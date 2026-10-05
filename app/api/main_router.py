import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from app.config import Settings, get_settings
from app.dependencies import get_engine, get_store
from app.ml_module.album import sync_album
from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import FaceEngine, InvalidImageError, decode_image
from app.schemas import (
    AlbumInfo,
    AlbumPhoto,
    BoundingBox,
    HealthResponse,
    MatchResponse,
    PhotoMatch,
    QueryFaceResult,
    SyncResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

# NOTE: the handlers below are plain `def`, not `async def`. Inference is CPU-bound,
# so FastAPI runs them in its threadpool instead of blocking the event loop.


def _bbox(values) -> BoundingBox:
    x1, y1, x2, y2 = (round(float(v), 1) for v in values)
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health(engine: FaceEngine = Depends(get_engine), store: EmbeddingStore = Depends(get_store)):
    return HealthResponse(
        status="ok",
        model_loaded=engine.is_loaded,
        album_ready=store.is_ready,
        photo_count=store.photo_count,
        face_count=store.face_count,
    )


@router.get("/album", response_model=AlbumInfo, tags=["Album"])
def album_info(store: EmbeddingStore = Depends(get_store)):
    """Photos currently indexed and how many faces each contains."""
    data = store.data
    if data is None:
        return AlbumInfo(photo_count=0, face_count=0, photos=[])
    return AlbumInfo(
        photo_count=store.photo_count,
        face_count=store.face_count,
        model_name=data.model_name,
        built_at=data.built_at,
        photos=[AlbumPhoto(filename=f, face_count=n) for f, n in sorted(data.faces_per_photo().items())],
    )


@router.post("/album/sync", response_model=SyncResponse, tags=["Album"])
def sync(
    full: bool = Query(False, description="Re-embed every photo instead of only new/changed ones"),
    engine: FaceEngine = Depends(get_engine),
    store: EmbeddingStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    """Update the .npz from the album directory: embed new/changed photos, drop deleted ones."""
    report = sync_album(engine, store, settings.album_dir, full=full)
    return SyncResponse(
        full_rebuild=report.full_rebuild,
        added=report.added,
        updated=report.updated,
        removed=report.removed,
        unchanged_count=report.unchanged_count,
        no_faces=report.no_faces,
        failed=report.failed,
        total_photos=report.total_photos,
        total_faces=report.total_faces,
        built_at=store.data.built_at if store.data else None,
    )


@router.post("/match", response_model=MatchResponse, tags=["Recognition"])
def match_face(
    request: Request,
    file: UploadFile = File(..., description="Image containing the face(s) to look up"),
    top_k: int | None = Query(None, ge=1, description="Max photos returned per face"),
    threshold: float | None = Query(None, ge=-1.0, le=1.0, description="Min cosine similarity to count as a match"),
    all_faces: bool = Query(False, description="Search for every detected face instead of only the largest"),
    engine: FaceEngine = Depends(get_engine),
    store: EmbeddingStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
):
    """Find every album photo that contains the person in the uploaded image."""
    top_k = min(top_k or settings.default_top_k, settings.max_top_k)
    threshold = settings.match_threshold if threshold is None else threshold

    if not store.is_ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Album is empty — call POST /album/sync first")
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

    results = []
    for i, face in enumerate(faces if all_faces else faces[:1]):
        hits = store.search_photos(face.embedding, threshold=threshold, top_k=top_k)
        results.append(QueryFaceResult(
            face_index=i,
            bbox=_bbox(face.bbox),
            det_score=round(face.det_score, 4),
            match_count=len(hits),
            matches=[
                PhotoMatch(
                    filename=hit.photo,
                    score=round(hit.score, 4),
                    face_bbox=_bbox(hit.bbox),
                    image_url=str(request.url_for("album-images", path=hit.photo)),
                )
                for hit in hits
            ],
        ))

    logger.info("Matched %d query face(s): %s", len(results), [r.match_count for r in results])
    return MatchResponse(faces_detected=len(faces), threshold=threshold, top_k=top_k, results=results)
