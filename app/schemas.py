from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    album_ready: bool
    photo_count: int
    face_count: int


class AlbumPhoto(BaseModel):
    filename: str
    face_count: int


class AlbumInfo(BaseModel):
    photo_count: int
    face_count: int
    model_name: str | None = None
    built_at: str | None = None
    photos: list[AlbumPhoto]


class SyncResponse(BaseModel):
    full_rebuild: bool
    added: list[str]
    updated: list[str]
    removed: list[str]
    unchanged_count: int
    no_faces: list[str] = Field(..., description="Newly processed photos in which no face was found")
    failed: dict[str, str] = Field(..., description="Photos that could not be read -> reason; retried next sync")
    total_photos: int
    total_faces: int
    built_at: str | None = None


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PhotoMatch(BaseModel):
    filename: str = Field(..., description="Album photo containing the person")
    score: float = Field(..., description="Cosine similarity of the best-matching face in the photo (max 1.0)")
    face_bbox: BoundingBox = Field(..., description="Where the matching face is in the album photo")
    image_url: str


class MatchResponse(BaseModel):
    matched: bool
    message: str
    threshold: float
    bbox: BoundingBox = Field(..., description="Face location in the uploaded image")
    det_score: float
    match_count: int
    matches: list[PhotoMatch]
