"""Album face embeddings persisted to an .npz file and searched with FAISS.

One row per *face* (a photo with n faces contributes n rows). A separate photo
registry records every processed photo, including ones with no faces, together
with its content hash so incremental syncs can skip unchanged files.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512


@dataclass
class AlbumData:
    """Everything stored in the .npz file."""

    embeddings: np.ndarray  # (F, 512) float32, L2-normalised
    face_photo: list[str]  # (F,) photo filename each face belongs to
    face_bbox: np.ndarray  # (F, 4) float32, x1 y1 x2 y2
    face_det_score: np.ndarray  # (F,) float32
    photo_hashes: dict[str, str] = field(default_factory=dict)  # every processed photo -> sha1
    model_name: str = ""
    built_at: str = ""

    @classmethod
    def empty(cls, model_name: str = "") -> "AlbumData":
        return cls(
            embeddings=np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
            face_photo=[],
            face_bbox=np.zeros((0, 4), dtype=np.float32),
            face_det_score=np.zeros((0,), dtype=np.float32),
            model_name=model_name,
        )

    @property
    def face_count(self) -> int:
        return len(self.face_photo)

    def faces_per_photo(self) -> dict[str, int]:
        counts = {photo: 0 for photo in self.photo_hashes}
        for photo in self.face_photo:
            counts[photo] = counts.get(photo, 0) + 1
        return counts


@dataclass(frozen=True)
class PhotoHit:
    photo: str
    score: float  # cosine similarity of the best-matching face in this photo
    bbox: tuple[float, float, float, float]  # that face's box


@dataclass(frozen=True)
class _Snapshot:
    data: AlbumData
    index: faiss.Index


class EmbeddingStore:
    """Holds the album in memory as a FAISS inner-product index.

    Embeddings are L2-normalised, so inner product == cosine similarity. A flat
    (exact) index is used: for tens of thousands of faces it is fast and never
    misses a match.

    Readers grab an immutable snapshot, so a sync can swap the index in without
    blocking or corrupting in-flight searches. Writers must hold `write_lock`.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._snapshot: _Snapshot | None = None
        self.write_lock = threading.Lock()

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def data(self) -> AlbumData | None:
        return self._snapshot.data if self._snapshot else None

    @property
    def is_ready(self) -> bool:
        return self._snapshot is not None and self._snapshot.index.ntotal > 0

    @property
    def face_count(self) -> int:
        return self._snapshot.index.ntotal if self._snapshot else 0

    @property
    def photo_count(self) -> int:
        return len(self._snapshot.data.photo_hashes) if self._snapshot else 0

    # ── persistence ──────────────────────────────────────────────────────────
    def load(self) -> bool:
        """Load the .npz from disk. Returns False if it does not exist yet."""
        if not self.path.exists():
            logger.warning("Embeddings file not found at %s — sync the album first", self.path)
            return False
        with np.load(self.path, allow_pickle=False) as npz:
            data = AlbumData(
                embeddings=npz["embeddings"].astype(np.float32),
                face_photo=[str(p) for p in npz["face_photo"]],
                face_bbox=npz["face_bbox"].astype(np.float32),
                face_det_score=npz["face_det_score"].astype(np.float32),
                photo_hashes=dict(zip((str(p) for p in npz["photos"]), (str(h) for h in npz["photo_hashes"]))),
                model_name=str(npz["model_name"]),
                built_at=str(npz["built_at"]),
            )
        self._swap(data)
        logger.info("Loaded %d faces from %d photos (%s)", self.face_count, self.photo_count, self.path)
        return True

    def save(self, data: AlbumData) -> None:
        """Persist to .npz and make it live for searches."""
        if not (len(data.embeddings) == len(data.face_photo) == len(data.face_bbox) == len(data.face_det_score)):
            raise ValueError("Per-face arrays must all be the same length")
        data.built_at = datetime.now(timezone.utc).isoformat()
        photos = sorted(data.photo_hashes)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename, so a crash never leaves a half-written file.
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez_compressed(
            tmp,
            embeddings=_as_matrix(data.embeddings).reshape(-1, EMBEDDING_DIM),
            face_photo=np.array(data.face_photo, dtype=np.str_),
            face_bbox=np.asarray(data.face_bbox, dtype=np.float32).reshape(-1, 4),
            face_det_score=np.asarray(data.face_det_score, dtype=np.float32),
            photos=np.array(photos, dtype=np.str_),
            photo_hashes=np.array([data.photo_hashes[p] for p in photos], dtype=np.str_),
            model_name=np.array(data.model_name),
            built_at=np.array(data.built_at),
        )
        tmp.replace(self.path)
        self._swap(data)
        logger.info("Saved %d faces from %d photos to %s", self.face_count, self.photo_count, self.path)

    def _swap(self, data: AlbumData) -> None:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if data.face_count:
            index.add(_as_matrix(data.embeddings))
        self._snapshot = _Snapshot(data=data, index=index)

    # ── search ───────────────────────────────────────────────────────────────
    def search_photos(self, query: np.ndarray, threshold: float, top_k: int) -> list[PhotoHit]:
        """Photos containing a face with cosine similarity >= threshold, best first.

        A photo appears once, scored by its best-matching face.
        """
        snapshot = self._snapshot
        if snapshot is None or snapshot.index.ntotal == 0:
            raise RuntimeError("Album is empty")

        # range_search returns *all* faces above the threshold, so no match is lost
        # to a top-k cut before de-duplicating by photo.
        _, scores, ids = snapshot.index.range_search(_as_matrix(query), threshold)

        best: dict[str, PhotoHit] = {}
        for score, idx in zip(scores, ids):
            photo = snapshot.data.face_photo[idx]
            if photo not in best or score > best[photo].score:
                bbox = tuple(float(v) for v in snapshot.data.face_bbox[idx])
                best[photo] = PhotoHit(photo=photo, score=float(score), bbox=bbox)
        return sorted(best.values(), key=lambda h: h.score, reverse=True)[:top_k]


def _as_matrix(embeddings: np.ndarray) -> np.ndarray:
    """FAISS wants a C-contiguous float32 2-D array."""
    arr = np.ascontiguousarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr
