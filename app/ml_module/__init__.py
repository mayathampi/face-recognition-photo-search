"""ML layer: face detection/embedding (InsightFace), gallery building and FAISS search.

Kept free of any FastAPI imports so it can be reused from scripts, workers or notebooks.
"""

from app.ml_module.embedding_store import EmbeddingStore, SearchHit
from app.ml_module.face_engine import DetectedFace, FaceEngine, InvalidImageError, decode_image, read_image
from app.ml_module.gallery import BuildReport, build_gallery, label_for, list_gallery_images

__all__ = [
    "BuildReport", "DetectedFace", "EmbeddingStore", "FaceEngine", "InvalidImageError", "SearchHit",
    "build_gallery", "decode_image", "label_for", "list_gallery_images", "read_image",
]
