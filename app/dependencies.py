from fastapi import Request

from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import FaceEngine


def get_engine(request: Request) -> FaceEngine:
    return request.app.state.engine


def get_store(request: Request) -> EmbeddingStore:
    return request.app.state.store
