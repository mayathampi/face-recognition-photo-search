import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.main_router import router
from app.config import get_settings
from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import FaceEngine

settings = get_settings()

settings.log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.log_dir / "app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_app(engine: FaceEngine | None = None, store: EmbeddingStore | None = None) -> FastAPI:
    """App factory. Tests pass in a fake engine/store; production builds the real ones at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            real_engine = FaceEngine(model_name=settings.model_name, ctx_id=settings.ctx_id,
                                     det_size=settings.det_size, min_det_score=settings.min_det_score,
                                     model_root=settings.model_root)
            real_engine.load()
            app.state.engine = real_engine
        if store is None:
            real_store = EmbeddingStore(settings.embeddings_path)
            real_store.load()
            app.state.store = real_store
        if app.state.store.model_name and app.state.store.model_name != app.state.engine.model_name:
            logger.warning("Gallery was built with '%s' but engine uses '%s' — rebuild the gallery",
                           app.state.store.model_name, app.state.engine.model_name)
        yield

    app = FastAPI(title=settings.app_name, version="1.0.0", root_path=settings.root_path, lifespan=lifespan)
    if engine is not None:
        app.state.engine = engine
    if store is not None:
        app.state.store = store

    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Browsers reject credentials with a wildcard origin.
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=settings.api_prefix)
    app.mount("/gallery-images", StaticFiles(directory=settings.gallery_dir, check_dir=False),
              name="gallery-images")

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": settings.app_name, "docs": "/docs"}

    return app


app = create_app()
