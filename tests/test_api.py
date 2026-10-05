"""API tests. A fake engine stands in for InsightFace, so no model download is needed."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import DetectedFace

DIM = 512


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEngine:
    """Returns a fixed face embedding chosen per test."""

    model_name = "fake"
    is_loaded = True

    def __init__(self):
        self.faces: list[DetectedFace] = []

    def detect(self, image):
        return self.faces


def face(embedding: np.ndarray) -> DetectedFace:
    return DetectedFace(embedding=embedding, bbox=(10.0, 10.0, 110.0, 110.0), det_score=0.99)


def png_bytes() -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def store(tmp_path):
    s = EmbeddingStore(tmp_path / "gallery.npz")
    s.save(np.stack([unit(1), unit(2), unit(3)]), ["alice/1.jpg", "bob.jpg", "carol.png"], "fake")
    return s


@pytest.fixture
def client(engine, store):
    with TestClient(create_app(engine=engine, store=store)) as c:
        yield c


def upload(client, **params):
    return client.post("/api/v1/match", params=params,
                       files={"file": ("q.png", png_bytes(), "image/png")})


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body == {"status": "ok", "model_loaded": True, "gallery_ready": True, "gallery_size": 3}


def test_gallery_info(client):
    body = client.get("/api/v1/gallery").json()
    assert body["size"] == 3
    assert body["filenames"] == ["alice/1.jpg", "bob.jpg", "carol.png"]


def test_match_returns_same_person_first(client, engine):
    # A slightly perturbed copy of bob's embedding should match bob.
    noisy = unit(2) + 0.05 * unit(99)
    engine.faces = [face(noisy / np.linalg.norm(noisy))]

    body = upload(client).json()
    matches = body["results"][0]["matches"]
    assert matches[0]["filename"] == "bob.jpg"
    assert matches[0]["label"] == "bob"
    assert matches[0]["score"] > 0.9
    assert matches[0]["image_url"].endswith("/gallery-images/bob.jpg")
    # Random unrelated embeddings sit near 0 cosine and are filtered by the threshold.
    assert len(matches) == 1


def test_match_label_from_folder(client, engine):
    engine.faces = [face(unit(1))]
    match = upload(client).json()["results"][0]["matches"][0]
    assert match["filename"] == "alice/1.jpg"
    assert match["label"] == "alice"


def test_match_threshold_and_top_k(client, engine):
    engine.faces = [face(unit(2))]
    body = upload(client, threshold=-1.0, top_k=2).json()
    assert len(body["results"][0]["matches"]) == 2


def test_all_faces(client, engine):
    engine.faces = [face(unit(1)), face(unit(3))]
    assert len(upload(client).json()["results"]) == 1
    body = upload(client, all_faces=True).json()
    assert [r["matches"][0]["filename"] for r in body["results"]] == ["alice/1.jpg", "carol.png"]


def test_no_face(client, engine):
    engine.faces = []
    assert upload(client).status_code == 422


def test_bad_image(client):
    r = client.post("/api/v1/match", files={"file": ("q.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_unsupported_type(client):
    r = client.post("/api/v1/match", files={"file": ("q.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_empty_gallery(engine, tmp_path):
    with TestClient(create_app(engine=engine, store=EmbeddingStore(tmp_path / "none.npz"))) as c:
        assert upload(c).status_code == 503


def test_store_roundtrip(store):
    reloaded = EmbeddingStore(store.path)
    assert reloaded.load()
    assert reloaded.filenames == store.filenames
    assert reloaded.model_name == "fake"
    hits = reloaded.search(unit(3), top_k=1)
    assert hits[0][0].filename == "carol.png"
