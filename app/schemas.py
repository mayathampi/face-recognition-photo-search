from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gallery_ready: bool
    gallery_size: int


class GalleryInfo(BaseModel):
    size: int
    model_name: str | None = None
    built_at: str | None = None
    filenames: list[str]


class RebuildResponse(BaseModel):
    indexed_count: int
    skipped_count: int
    skipped: dict[str, str] = Field(default_factory=dict, description="filename -> reason it was skipped")
    built_at: str | None = None


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Match(BaseModel):
    filename: str = Field(..., description="Path of the gallery image, relative to the gallery directory")
    label: str = Field(..., description="Person label: parent folder name, or file stem for flat galleries")
    score: float = Field(..., description="Cosine similarity, higher is more similar (max 1.0)")
    image_url: str


class FaceMatchResult(BaseModel):
    face_index: int
    bbox: BoundingBox
    det_score: float
    matches: list[Match]


class MatchResponse(BaseModel):
    faces_detected: int
    threshold: float
    top_k: int
    results: list[FaceMatchResult]
