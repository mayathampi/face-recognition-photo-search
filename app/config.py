from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings, overridable via environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Face Recognition API"
    # Set when running behind a reverse proxy that strips a path prefix (e.g. /backend).
    root_path: str = ""
    # Comma-separated list of allowed origins, or "*".
    cors_origins: str = "*"
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"

    # Album: one flat directory of photos, and the .npz holding an embedding per face.
    album_dir: Path = BASE_DIR / "data" / "album"
    embeddings_path: Path = BASE_DIR / "data" / "embeddings" / "album.npz"

    # InsightFace model pack. buffalo_l = RetinaFace detector + ArcFace (ResNet50) recognizer.
    model_name: str = "buffalo_l"
    # Where model packs are cached/downloaded.
    model_root: Path = Path.home() / ".insightface"
    # -1 runs on CPU; 0, 1, ... selects a CUDA device (requires onnxruntime-gpu).
    ctx_id: int = -1
    det_size: int = 640
    # Faces detected below this confidence are ignored.
    min_det_score: float = 0.5

    # Cosine similarity cut-off for a "match". On the sample album, true matches score >= 0.505
    # and the closest wrong match 0.458, so 0.50 keeps every true match with no false ones.
    match_threshold: float = 0.50
    max_upload_mb: int = 10

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
