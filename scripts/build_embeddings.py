"""Build data/embeddings/gallery.npz from the images in data/images.

Usage:
    python -m scripts.build_embeddings
    python -m scripts.build_embeddings --gallery-dir /path/to/images --output /path/to/gallery.npz
"""

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import FaceEngine
from app.ml_module.gallery import build_gallery


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery-dir", type=Path, default=settings.gallery_dir)
    parser.add_argument("--output", type=Path, default=settings.embeddings_path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    engine = FaceEngine(model_name=settings.model_name, ctx_id=settings.ctx_id,
                        det_size=settings.det_size, min_det_score=settings.min_det_score,
                        model_root=settings.model_root)
    engine.load()
    store = EmbeddingStore(args.output)

    try:
        report = build_gallery(engine, store, args.gallery_dir)
    except ValueError as exc:
        logging.error(str(exc))
        return 1

    print(f"\nIndexed {len(report.indexed)} image(s) -> {args.output}")
    if report.skipped:
        print(f"Skipped {len(report.skipped)}:")
        for name, reason in report.skipped.items():
            print(f"  - {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
