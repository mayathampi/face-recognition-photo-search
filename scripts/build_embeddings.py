"""Sync data/embeddings/album.npz with the photos in data/album.

Run it whenever photos are added, changed or removed: only new/changed photos are
embedded, deleted photos are dropped, everything else is reused.

Usage:
    python -m scripts.build_embeddings
    python -m scripts.build_embeddings --full                # re-embed every photo
    python -m scripts.build_embeddings --album-dir /path/to/photos --output /path/to/album.npz
"""

import argparse
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.ml_module.album import sync_album
from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import FaceEngine


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--album-dir", type=Path, default=settings.album_dir)
    parser.add_argument("--output", type=Path, default=settings.embeddings_path)
    parser.add_argument("--full", action="store_true", help="re-embed every photo")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.album_dir.is_dir():
        logging.error("Album directory not found: %s", args.album_dir)
        return 1

    engine = FaceEngine(model_name=settings.model_name, ctx_id=settings.ctx_id,
                        det_size=settings.det_size, min_det_score=settings.min_det_score,
                        model_root=settings.model_root)
    store = EmbeddingStore(args.output)
    store.load()
    # The model loads lazily on the first detect(), so an unchanged album syncs instantly.

    report = sync_album(engine, store, args.album_dir, full=args.full)

    print(f"\n{'Full rebuild' if report.full_rebuild else 'Sync'} -> {args.output}")
    print(f"  added:     {len(report.added)}")
    print(f"  updated:   {len(report.updated)}")
    print(f"  removed:   {len(report.removed)}")
    print(f"  unchanged: {report.unchanged_count}")
    print(f"  total:     {report.total_photos} photos, {report.total_faces} faces")
    if report.no_faces:
        print(f"  no faces found in: {', '.join(report.no_faces)}")
    for name, reason in report.failed.items():
        print(f"  FAILED {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
