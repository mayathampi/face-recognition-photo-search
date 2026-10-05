"""Builds the gallery: walks the images directory and extracts one embedding per image."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import FaceEngine, InvalidImageError, read_image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BuildReport:
    indexed: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # filename -> reason


def list_gallery_images(gallery_dir: Path) -> list[Path]:
    """Image files under gallery_dir, recursively (person-per-folder layouts work too)."""
    if not gallery_dir.is_dir():
        return []
    return sorted(p for p in gallery_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def label_for(filename: str) -> str:
    """'alice/01.jpg' -> 'alice'; 'bob.jpg' -> 'bob'."""
    path = Path(filename)
    return path.parent.name if path.parent.name else path.stem


def build_gallery(engine: FaceEngine, store: EmbeddingStore, gallery_dir: Path) -> BuildReport:
    """Embed every gallery image (largest face per image) and save to the store."""
    report = BuildReport()
    embeddings: list[np.ndarray] = []

    images = list_gallery_images(gallery_dir)
    logger.info("Building gallery from %d images in %s", len(images), gallery_dir)

    for path in images:
        rel = path.relative_to(gallery_dir).as_posix()
        try:
            faces = engine.detect(read_image(path))
        except InvalidImageError as exc:
            report.skipped[rel] = str(exc)
            continue
        if not faces:
            report.skipped[rel] = "No face detected"
            continue
        if len(faces) > 1:
            logger.info("%s has %d faces — using the largest", rel, len(faces))
        embeddings.append(faces[0].embedding)
        report.indexed.append(rel)

    if not embeddings:
        raise ValueError(f"No usable faces found in {gallery_dir}")

    store.save(np.stack(embeddings), report.indexed, engine.model_name)
    for rel, reason in report.skipped.items():
        logger.warning("Skipped %s: %s", rel, reason)
    return report
