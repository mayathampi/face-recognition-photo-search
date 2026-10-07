"""API + album sync tests. A fake engine stands in for InsightFace, so no model download is needed.

Test photos are tiny solid-colour PNGs; the fake engine reads the colour value and returns
the faces registered for it, so each "photo" can contain any set of people.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.ml_module.album import sync_album
from app.ml_module.embedding_store import EmbeddingStore
from app.ml_module.face_engine import DetectedFace

DIM = 512


def person(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


ALICE, BOB, CAROL, DAVE = person(1), person(2), person(3), person(4)


def face(embedding: np.ndarray, size: float = 100.0) -> DetectedFace:
    return DetectedFace(embedding=embedding, bbox=(10.0, 10.0, 10.0 + size, 10.0 + size), det_score=0.99)


class FakeEngine:
    model_name = "fake"
    is_loaded = True

    def __init__(self):
        self.by_colour: dict[int, list[DetectedFace]] = {}
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.by_colour.get(int(image[0, 0, 0]), [])


def png(colour: int) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((16, 16, 3), colour, dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def album_dir(tmp_path, monkeypatch):
    d = tmp_path / "album"
    d.mkdir()
    monkeypatch.setattr(get_settings(), "album_dir", d)
    return d


@pytest.fixture
def engine():
    e = FakeEngine()
    e.by_colour = {
        10: [face(ALICE, 120), face(BOB, 80)],  # group photo: alice + bob
        20: [face(BOB)],  # bob alone
        30: [face(ALICE), face(CAROL), face(BOB, 50)],  # alice, carol, bob
        40: [],  # landscape, no faces
        50: [face(DAVE)],  # query-only person, never in the album
    }
    return e


@pytest.fixture
def store(tmp_path):
    return EmbeddingStore(tmp_path / "album.npz")


@pytest.fixture
def populated(album_dir, engine, store):
    (album_dir / "party.png").write_bytes(png(10))
    (album_dir / "bob.png").write_bytes(png(20))
    (album_dir / "trip.png").write_bytes(png(30))
    (album_dir / "beach.png").write_bytes(png(40))
    (album_dir / "notes.txt").write_text("ignored")
    sync_album(engine, store, album_dir)
    return store


@pytest.fixture
def client(engine, populated):
    with TestClient(create_app(engine=engine, store=populated)) as c:
        yield c


def query(client, colour: int, **params):
    return client.post("/match", params=params, files={"file": ("q.png", png(colour), "image/png")})


# ── album sync ───────────────────────────────────────────────────────────────

def test_sync_stores_every_face(populated):
    data = populated.data
    assert populated.photo_count == 4  # beach.png has no faces but is still registered
    assert populated.face_count == 6  # 2 + 1 + 3 + 0
    assert data.faces_per_photo() == {"party.png": 2, "bob.png": 1, "trip.png": 3, "beach.png": 0}


def test_sync_is_incremental(album_dir, engine, populated):
    engine.calls = 0
    report = sync_album(engine, populated, album_dir)
    assert engine.calls == 0
    assert not report.changed and report.unchanged_count == 4

    (album_dir / "new.png").write_bytes(png(20))  # add
    (album_dir / "bob.png").write_bytes(png(10))  # change content
    (album_dir / "trip.png").unlink()  # delete
    report = sync_album(engine, populated, album_dir)

    assert engine.calls == 2  # only new.png and bob.png were embedded
    assert report.added == ["new.png"]
    assert report.updated == ["bob.png"]
    assert report.removed == ["trip.png"]
    assert report.unchanged_count == 2
    assert populated.data.faces_per_photo() == {"party.png": 2, "bob.png": 2, "beach.png": 0, "new.png": 1}


def test_sync_persists_to_npz(album_dir, engine, populated):
    reloaded = EmbeddingStore(populated.path)
    assert reloaded.load()
    assert reloaded.data.faces_per_photo() == populated.data.faces_per_photo()
    # Reloaded store is still incremental.
    engine.calls = 0
    sync_album(engine, reloaded, album_dir)
    assert engine.calls == 0


def test_full_sync_reembeds_everything(album_dir, engine, populated):
    engine.calls = 0
    report = sync_album(engine, populated, album_dir, full=True)
    assert report.full_rebuild and engine.calls == 4
    assert populated.face_count == 6


def test_undecodable_photo_is_reported_and_retried(album_dir, engine, populated):
    (album_dir / "broken.jpg").write_bytes(b"not an image")
    report = sync_album(engine, populated, album_dir)
    assert "broken.jpg" in report.failed
    assert "broken.jpg" not in populated.data.photo_hashes
    assert "broken.jpg" in sync_album(engine, populated, album_dir).failed


# ── API ──────────────────────────────────────────────────────────────────────

def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok", "model_loaded": True, "album_ready": True, "photo_count": 4, "face_count": 6,
    }


def test_album_info(client):
    body = client.get("/album").json()
    assert body["face_count"] == 6
    assert {p["filename"]: p["face_count"] for p in body["photos"]}["trip.png"] == 3


def test_match_returns_every_photo_with_the_person(client):
    body = query(client, 20).json()  # bob
    matches = body["matches"]
    assert sorted(m["filename"] for m in matches) == ["bob.png", "party.png", "trip.png"]
    assert body["matched"] is True and body["match_count"] == 3
    assert all(m["score"] > 0.99 for m in matches)
    trip = next(m for m in matches if m["filename"] == "trip.png")
    assert trip["face_bbox"] == {"x1": 10.0, "y1": 10.0, "x2": 60.0, "y2": 60.0}  # bob's face in trip.png
    assert trip["image_url"].endswith("/album-images/trip.png")


def test_match_unknown_person_returns_nothing(client):
    body = query(client, 50).json()
    assert body["matched"] is False and body["matches"] == []
    assert body["message"] == "No matching photos found in the album"


def test_match_uses_configured_threshold(client):
    assert query(client, 20).json()["threshold"] == get_settings().match_threshold


def test_match_rejects_multiple_faces(client):
    r = query(client, 10)  # alice + bob
    assert r.status_code == 422
    assert "2 faces" in r.json()["detail"]


def test_sync_endpoint(client, album_dir):
    (album_dir / "extra.png").write_bytes(png(30))
    body = client.post("/album/sync").json()
    assert body["added"] == ["extra.png"]
    assert body["total_faces"] == 9


def test_no_face(client):
    assert query(client, 40).status_code == 422


def test_bad_image(client):
    r = client.post("/match", files={"file": ("q.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_unsupported_type(client):
    r = client.post("/match", files={"file": ("q.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/match", files={"file": ("q.webp", png(20), "image/webp")})
    assert r.status_code == 415


def test_extension_is_case_insensitive_and_content_type_ignored(client):
    r = client.post("/match", files={"file": ("Q.JPEG", png(20), "application/octet-stream")})
    assert r.status_code == 200


def test_empty_album(engine, tmp_path):
    with TestClient(create_app(engine=engine, store=EmbeddingStore(tmp_path / "none.npz"))) as c:
        assert query(c, 20).status_code == 503
