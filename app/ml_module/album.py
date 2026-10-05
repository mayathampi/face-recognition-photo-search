"""Album sync: keeps the .npz in step with the photos in the album directory.

Every face in every photo is embedded (a photo with n faces adds n rows). Syncs are
incremental: photos are identified by a SHA-1 of their bytes, so only new or changed
photos go through the model, deleted photos are dropped, and the rest are reused.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ml_module.embedding_store import AlbumData, EmbeddingStore
from app.ml_module.face_engine import FaceEngine, InvalidImageError, decode_image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SyncReport:
    full_rebuild: bool = False
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged_count: int = 0
    no_faces: list[str] = field(default_factory=list)  # newly processed photos with zero faces
    failed: dict[str, str] = field(default_factory=dict)  # photo -> reason (not stored; retried next sync)
    total_photos: int = 0
    total_faces: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def list_album_photos(album_dir: Path) -> list[Path]:
    """Image files directly inside album_dir (hidden files ignored)."""
    if not album_dir.is_dir():
        return []
    return sorted(
        p for p in album_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sync_album(engine: FaceEngine, store: EmbeddingStore, album_dir: Path, full: bool = False) -> SyncReport:
    """Bring the store up to date with album_dir and save it if anything changed."""
    with store.write_lock:
        old = store.data
        if old is not None and old.model_name != engine.model_name:
            logger.warning("Album was embedded with '%s', engine is '%s' — doing a full rebuild",
                           old.model_name, engine.model_name)
            full = True
        if old is None or full:
            old = AlbumData.empty(engine.model_name)

        report = SyncReport(full_rebuild=full)
        new_hashes: dict[str, str] = {}
        to_process: list[tuple[str, bytes]] = []

        for path in list_album_photos(album_dir):
            name = path.name
            raw = path.read_bytes()
            digest = hashlib.sha1(raw).hexdigest()
            previous = old.photo_hashes.get(name)
            if previous == digest:
                new_hashes[name] = digest
                report.unchanged_count += 1
                continue
            (report.updated if previous else report.added).append(name)
            to_process.append((name, raw))
            new_hashes[name] = digest

        report.removed = sorted(set(old.photo_hashes) - set(new_hashes))

        # Keep the stored faces of photos that did not change.
        unchanged = {n for n, h in new_hashes.items() if old.photo_hashes.get(n) == h}
        keep = [i for i, photo in enumerate(old.face_photo) if photo in unchanged]
        embeddings = [old.embeddings[keep]]
        face_photo = [old.face_photo[i] for i in keep]
        face_bbox = [old.face_bbox[keep]]
        face_det_score = [old.face_det_score[keep]]

        for i, (name, raw) in enumerate(to_process, 1):
            try:
                faces = engine.detect(decode_image(raw))
            except InvalidImageError as exc:
                report.failed[name] = str(exc)
                del new_hashes[name]
                continue
            logger.info("[%d/%d] %s: %d face(s)", i, len(to_process), name, len(faces))
            if not faces:
                report.no_faces.append(name)
                continue
            embeddings.append(np.stack([f.embedding for f in faces]))
            face_photo.extend([name] * len(faces))
            face_bbox.append(np.array([f.bbox for f in faces], dtype=np.float32))
            face_det_score.append(np.array([f.det_score for f in faces], dtype=np.float32))

        # A photo that failed to decode is not stored, so it doesn't count as added/updated.
        report.added = [n for n in report.added if n not in report.failed]
        report.updated = [n for n in report.updated if n not in report.failed]

        data = AlbumData(
            embeddings=np.concatenate(embeddings).astype(np.float32),
            face_photo=face_photo,
            face_bbox=np.concatenate(face_bbox).astype(np.float32),
            face_det_score=np.concatenate(face_det_score).astype(np.float32),
            photo_hashes=new_hashes,
            model_name=engine.model_name,
        )
        report.total_photos = len(new_hashes)
        report.total_faces = data.face_count

        if report.changed or full or not store.path.exists():
            store.save(data)
        else:
            logger.info("Album unchanged (%d photos, %d faces) — nothing to save",
                        report.total_photos, report.total_faces)

    for name, reason in report.failed.items():
        logger.warning("Failed %s: %s", name, reason)
    return report
