"""Face detection + embedding extraction, backed by InsightFace (RetinaFace + ArcFace)."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class InvalidImageError(ValueError):
    """Raised when bytes cannot be decoded into an image."""


@dataclass(frozen=True)
class DetectedFace:
    embedding: np.ndarray  # (512,) float32, L2-normalised
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    det_score: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/...) into a BGR ndarray."""
    if not data:
        raise InvalidImageError("Empty image")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("Could not decode image")
    return image


def read_image(path: Path) -> np.ndarray:
    """Read an image from disk. Uses imdecode so non-ASCII paths work too."""
    return decode_image(path.read_bytes())


class FaceEngine:
    """Thin wrapper around insightface.app.FaceAnalysis.

    The model pack is downloaded to model_root (default ~/.insightface) on first load.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = -1, det_size: int = 640,
                 min_det_score: float = 0.5, model_root: Path | None = None):
        self.model_name = model_name
        self.model_root = Path(model_root or Path.home() / ".insightface")
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.min_det_score = min_det_score
        self._app = None

    @property
    def is_loaded(self) -> bool:
        return self._app is not None

    def load(self) -> None:
        if self._app is not None:
            return
        # Imported lazily so the API (and tests) can start without pulling in the model stack.
        from insightface.app import FaceAnalysis

        providers = ["CPUExecutionProvider"] if self.ctx_id < 0 else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Loading InsightFace model '%s' (ctx_id=%s)", self.model_name, self.ctx_id)
        app = FaceAnalysis(name=self.model_name, root=str(self.model_root), providers=providers,
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=self.ctx_id, det_size=(self.det_size, self.det_size))
        self._app = app
        logger.info("InsightFace model loaded")

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect every face in a BGR image, largest first."""
        self.load()
        raw = self._app.get(image)
        offset = 0
        if not raw:
            # Tightly-cropped faces (passport/ID style) fill the frame and the detector misses
            # them. Retry with a border so the face looks like it sits in a normal photo.
            offset = max(image.shape[:2]) // 2
            padded = cv2.copyMakeBorder(image, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=0)
            raw = self._app.get(padded)

        faces = []
        for f in raw:
            if float(f.det_score) < self.min_det_score:
                continue
            faces.append(DetectedFace(
                embedding=np.asarray(f.normed_embedding, dtype=np.float32),
                bbox=tuple(float(v) - offset for v in f.bbox),
                det_score=float(f.det_score),
            ))
        faces.sort(key=lambda face: face.area, reverse=True)
        return faces
