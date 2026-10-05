"""ML layer: face detection/embedding (InsightFace), album sync and FAISS search.

Kept free of any FastAPI imports so it can be reused from scripts, workers or notebooks.
"""

from app.ml_module.album import SyncReport, list_album_photos, sync_album
from app.ml_module.embedding_store import AlbumData, EmbeddingStore, PhotoHit
from app.ml_module.face_engine import DetectedFace, FaceEngine, InvalidImageError, decode_image, read_image

__all__ = [
    "AlbumData", "DetectedFace", "EmbeddingStore", "FaceEngine", "InvalidImageError", "PhotoHit", "SyncReport",
    "decode_image", "list_album_photos", "read_image", "sync_album",
]
