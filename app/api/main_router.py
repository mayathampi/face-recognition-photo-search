import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from app.config import Settings, get_settings
from app.dependencies import get_engine, get_store
from app.schemas import (
    BoundingBox,
    FaceMatchResult,
    GalleryInfo,
    HealthResponse,
    Match,
    MatchResponse,
    RebuildResponse,
)
from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import FaceEngine, InvalidImageError, decode_image
from app.ml_module.gallery import build_gallery, label_for

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

# NOTE: the handlers below are plain `def`, not `async def`. Inference is CPU-bound,
# so FastAPI runs them in its threadpool instead of blocking the event loop.


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(engine: FaceEngine = Depends(get_engine), store: EmbeddingStore = Depends(get_store)):
    return HealthResponse(
        status="ok",
        model_loaded=engine.is_loaded,
        gallery_ready=store.is_ready,
        gallery_size=store.size,
    )


@router.get("/gallery", response_model=GalleryInfo, tags=["Gallery"])
def gallery_info(store: EmbeddingStore = Depends(get_store)):
    """List the images currently indexed in the gallery."""
    return GalleryInfo(size=store.size, model_name=store.model_name,
                       built_at=store.built_at, filenames=store.filenames)


@router.post("/gallery/rebuild", response_model=RebuildResponse, tags=["Gallery"])
def rebuild_gallery(
    engine: FaceEngine = Depends(get_engine),
    store: EmbeddingStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    """Re-scan the gallery directory, re-embed every image and overwrite the .npz file."""
    try:
        report = build_gallery(engine, store, settings.gallery_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return RebuildResponse(
        indexed_count=len(report.indexed),
        skipped_count=len(report.skipped),
        skipped=report.skipped,
        built_at=store.built_at,
    )


@router.post("/match", response_model=MatchResponse, tags=["Recognition"])
def match_face(
    request: Request,
    file: UploadFile = File(..., description="Image containing the face(s) to look up"),
    top_k: int | None = Query(None, ge=1, description="Max matches returned per face"),
    threshold: float | None = Query(None, ge=-1.0, le=1.0, description="Min cosine similarity to count as a match"),
    all_faces: bool = Query(False, description="Match every detected face instead of only the largest"),
    engine: FaceEngine = Depends(get_engine),
    store: EmbeddingStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    """Compare the uploaded face against the gallery and return the matching images."""
    top_k = min(top_k or settings.default_top_k, settings.max_top_k)
    threshold = settings.match_threshold if threshold is None else threshold

    if not store.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Gallery is empty — call POST /gallery/rebuild first")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=f"Unsupported content type '{file.content_type}'")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Image exceeds {settings.max_upload_mb} MB")

    try:
        image = decode_image(data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    faces = engine.detect(image)
    if not faces:
        raise HTTPException(status_code=422, detail="No face detected in image")

    query_faces = faces if all_faces else faces[:1]
    hits_per_face = store.search([f.embedding for f in query_faces], top_k=top_k, threshold=threshold)

    results = []
    for i, (face, hits) in enumerate(zip(query_faces, hits_per_face)):
        results.append(FaceMatchResult(
            face_index=i,
            bbox=BoundingBox(x1=face.bbox[0], y1=face.bbox[1], x2=face.bbox[2], y2=face.bbox[3]),
            det_score=face.det_score,
            matches=[
                Match(
                    filename=hit.filename,
                    label=label_for(hit.filename),
                    score=round(hit.score, 4),
                    image_url=str(request.url_for("gallery-images", path=hit.filename)),
                )
                for hit in hits
            ],
        ))

    logger.info("Matched %d face(s); best score %s", len(results),
                results[0].matches[0].score if results[0].matches else None)
    return MatchResponse(faces_detected=len(faces), threshold=threshold, top_k=top_k, results=results)
