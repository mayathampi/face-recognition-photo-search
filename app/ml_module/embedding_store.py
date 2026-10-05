"""Gallery embeddings persisted to an .npz file and searched with FAISS."""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    filename: str
    score: float  # cosine similarity in [-1, 1]


@dataclass(frozen=True)
class _Snapshot:
    index: faiss.Index
    filenames: list[str]
    model_name: str
    built_at: str


class EmbeddingStore:
    """Holds the gallery in memory as a FAISS inner-product index.

    Embeddings are L2-normalised, so inner product == cosine similarity. A flat
    (exact) index is used: for a few thousand faces it is faster than any
    approximate index and never misses a match.

    Readers grab an immutable snapshot, so a rebuild can swap the index in
    without blocking or corrupting in-flight searches.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._snapshot: _Snapshot | None = None
        self._write_lock = threading.Lock()

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return self._snapshot is not None and self._snapshot.index.ntotal > 0

    @property
    def size(self) -> int:
        return self._snapshot.index.ntotal if self._snapshot else 0

    @property
    def filenames(self) -> list[str]:
        return list(self._snapshot.filenames) if self._snapshot else []

    @property
    def model_name(self) -> str | None:
        return self._snapshot.model_name if self._snapshot else None

    @property
    def built_at(self) -> str | None:
        return self._snapshot.built_at if self._snapshot else None

    # ── persistence ──────────────────────────────────────────────────────────
    def load(self) -> bool:
        """Load the .npz from disk. Returns False if it does not exist yet."""
        if not self.path.exists():
            logger.warning("Embeddings file not found at %s — build the gallery first", self.path)
            return False
        with np.load(self.path, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            filenames = [str(f) for f in data["filenames"]]
            model_name = str(data["model_name"])
            built_at = str(data["built_at"])
        self._swap(embeddings, filenames, model_name, built_at)
        logger.info("Loaded %d gallery embeddings from %s", self.size, self.path)
        return True

    def save(self, embeddings: np.ndarray, filenames: list[str], model_name: str) -> None:
        """Persist embeddings to .npz and make them live for searches."""
        if len(embeddings) != len(filenames):
            raise ValueError("embeddings and filenames must be the same length")
        built_at = datetime.now(timezone.utc).isoformat()
        embeddings = _as_matrix(embeddings)
        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file then rename, so a crash never leaves a half-written gallery.
            tmp = self.path.with_name(self.path.stem + ".tmp.npz")
            np.savez_compressed(
                tmp,
                embeddings=embeddings,
                filenames=np.array(filenames, dtype=np.str_),
                model_name=np.array(model_name),
                built_at=np.array(built_at),
            )
            tmp.replace(self.path)
            self._swap(embeddings, filenames, model_name, built_at)
        logger.info("Saved %d gallery embeddings to %s", len(filenames), self.path)

    def _swap(self, embeddings: np.ndarray, filenames: list[str], model_name: str, built_at: str) -> None:
        embeddings = _as_matrix(embeddings)
        dim = embeddings.shape[1] if embeddings.size else 512
        index = faiss.IndexFlatIP(dim)
        if embeddings.size:
            index.add(embeddings)
        self._snapshot = _Snapshot(index=index, filenames=list(filenames),
                                   model_name=model_name, built_at=built_at)

    # ── search ───────────────────────────────────────────────────────────────
    def search(self, queries: np.ndarray, top_k: int, threshold: float | None = None) -> list[list[SearchHit]]:
        """Return the top_k gallery hits for each query embedding, best first."""
        snapshot = self._snapshot
        if snapshot is None or snapshot.index.ntotal == 0:
            raise RuntimeError("Gallery is empty")
        queries = _as_matrix(queries)
        k = min(top_k, snapshot.index.ntotal)
        scores, ids = snapshot.index.search(queries, k)

        results = []
        for row_scores, row_ids in zip(scores, ids):
            hits = []
            for score, idx in zip(row_scores, row_ids):
                if idx < 0 or (threshold is not None and score < threshold):
                    continue
                hits.append(SearchHit(filename=snapshot.filenames[idx], score=float(score)))
            results.append(hits)
        return results


def _as_matrix(embeddings: np.ndarray) -> np.ndarray:
    """FAISS wants a C-contiguous float32 2-D array."""
    arr = np.ascontiguousarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr
