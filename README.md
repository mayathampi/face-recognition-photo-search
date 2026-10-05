# Facerecognition

A FastAPI service for face search. You put a folder of reference images (the gallery) in place, it stores one face embedding per image in an `.npz` file, and an API endpoint takes an uploaded photo and returns the gallery images that show the same person.

- **Detection + embeddings:** [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (RetinaFace detector + ArcFace ResNet50 recognizer, 512-d embeddings) on ONNX Runtime
- **Search:** [FAISS](https://github.com/facebookresearch/faiss) `IndexFlatIP` over L2-normalised embeddings, so the score is cosine similarity
- **Storage:** `data/embeddings/gallery.npz` (`embeddings`, `filenames`, `model_name`, `built_at`)

## Project layout

```
app/
├── main.py              # app factory, lifespan (loads model + gallery), CORS, static mount
├── config.py            # pydantic-settings; every field can be overridden by env / .env
├── schemas.py           # request/response models
├── dependencies.py      # FastAPI dependencies (engine, store)
├── api/
│   └── main_router.py   # all endpoints
└── ml_module/           # ML layer, no FastAPI imports
    ├── face_engine.py      # InsightFace wrapper: decode image, detect faces, embed
    ├── embedding_store.py  # .npz persistence + FAISS index (thread-safe hot swap)
    └── gallery.py          # walks the images dir, one embedding per image
scripts/
└── build_embeddings.py  # CLI that builds gallery.npz
tests/
└── test_api.py          # API tests with a fake engine (no model download needed)
data/
├── images/              # put your gallery images here (git-ignored)
└── embeddings/          # gallery.npz is written here (git-ignored)
```

## Quick start (local)

Requires Python 3.10+ and a C compiler. InsightFace builds a small C extension on install; on Ubuntu, run `sudo apt install build-essential python3-dev`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

# 1. Add your images: data/images/alice.jpg, or one folder per person: data/images/alice/1.jpg
# 2. Build the embeddings. The first run downloads the model (~280 MB) to ~/.insightface
python -m scripts.build_embeddings

# 3. Run the API
uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs

## Docker

The model pack is baked into the image at build time, so the container starts without downloading anything.

```bash
docker compose up --build
# rebuild the embeddings after adding images to ./data/images
curl -X POST localhost:8000/api/v1/gallery/rebuild
```

Without compose:

```bash
docker build -t facerecognition .
docker run -p 8000:8000 -v "$PWD/data:/app/data" facerecognition
```

## API

| Method | Path | Description |
|---|---|---|
| GET  | `/api/v1/health` | Whether the model is loaded and the gallery is ready |
| GET  | `/api/v1/gallery` | Indexed filenames, model name, build time |
| POST | `/api/v1/gallery/rebuild` | Re-scan `data/images`, re-embed every image, overwrite the `.npz` |
| POST | `/api/v1/match` | Upload an image (multipart field `file`) and get the matching gallery images |
| GET  | `/gallery-images/{filename}` | Serves a gallery image (used for `image_url` in match results) |

`/match` query params: `top_k` (default 5), `threshold` (default 0.45), `all_faces` (default `false`, which matches only the largest face).

```bash
curl -F "file=@query.jpg" "localhost:8000/api/v1/match?top_k=3"
```

```json
{
  "faces_detected": 1,
  "threshold": 0.45,
  "top_k": 3,
  "results": [{
    "face_index": 0,
    "bbox": {"x1": 146.5, "y1": 156.0, "x2": 421.6, "y2": 491.3},
    "det_score": 0.84,
    "matches": [
      {"filename": "alice/1.jpg", "label": "alice", "score": 0.9447,
       "image_url": "http://localhost:8000/gallery-images/alice/1.jpg"}
    ]
  }]
}
```

Errors: `400` for an undecodable image, `413` for a file over `MAX_UPLOAD_MB`, `415` for an unsupported type, `422` when no face is detected, `503` when the gallery is empty.

## Tuning

- **`MATCH_THRESHOLD`** (cosine similarity): with ArcFace, the same person usually scores **0.5–0.9** and different people score **< 0.3**. Raise it for fewer false positives; lower it if real matches are being missed.
- **Gallery images**: when an image contains several faces, the largest one is used. Tightly cropped faces are handled by padding the image and retrying detection.
- **GPU**: install `onnxruntime-gpu` instead of `onnxruntime` and set `CTX_ID=0`.
- **Scale**: the flat index does an exact search and is fast up to roughly 100k faces. Beyond that, switch `EmbeddingStore` to `faiss.IndexHNSWFlat`.

## Tests

```bash
pytest -q
```

## Push to GitHub

```bash
git remote add origin git@github.com:<your-account>/Facerecognition.git
git push -u origin main
```
