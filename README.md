# Facerecognition

A FastAPI service for finding a person across a photo album. Put all your photos in one folder. A sync script detects **every face in every photo** and stores one embedding per face in an `.npz` file. Then upload a photo of someone, and the API returns every album photo that contains them.

- **Detection + embeddings:** [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (RetinaFace detector + ArcFace ResNet50 recognizer, 512-d embeddings) on ONNX Runtime
- **Search:** [FAISS](https://github.com/facebookresearch/faiss) `IndexFlatIP` over L2-normalised embeddings, so the score is cosine similarity. `range_search` returns every face above the threshold, and the results are grouped by photo.
- **Storage:** `data/embeddings/album.npz`, one row per face
- **Incremental sync:** photos are tracked by SHA-1, so each sync embeds only new or changed photos and drops deleted ones

## Project layout

```
app/
├── main.py              # app factory, lifespan (loads model + album), CORS, static mount
├── config.py            # pydantic-settings; every field can be overridden by env / .env
├── schemas.py           # request/response models
├── dependencies.py      # FastAPI dependencies (engine, store)
├── api/
│   └── main_router.py   # all endpoints
└── ml_module/           # ML layer, no FastAPI imports
    ├── face_engine.py      # InsightFace wrapper: decode image, detect faces, embed
    ├── embedding_store.py  # .npz persistence + FAISS index + photo-level search
    └── album.py            # incremental album sync (every face of every photo)
scripts/
├── build_embeddings.py          # CLI: sync the album into album.npz
└── download_celebrity_album.py  # builds a sample album from Wikimedia Commons
tests/
└── test_api.py          # sync + API tests with a fake engine (no model download needed)
data/                    # all git-ignored
├── album/               # your photos, one flat folder
├── embeddings/          # album.npz
└── queries/             # sample query portraits (from the download script)
```

## Quick start (local)

Requires Python 3.10+ and a C compiler. InsightFace builds a small C extension on install; on Ubuntu, run `sudo apt install build-essential python3-dev`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

# 1. Put photos in data/album/, or build the sample celebrity album:
python -m scripts.download_celebrity_album

# 2. Sync the embeddings. The first run downloads the model (~280 MB) to ~/.insightface.
#    Re-run it any time photos are added, changed or removed.
python -m scripts.build_embeddings

# 3. Run the API
uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs

### What a sync does

| Photo state | Action |
|---|---|
| New | Detect all faces and add one row per face |
| Content changed (different hash) | Drop its old rows, re-detect, add new rows |
| Deleted from the folder | Drop its rows |
| Unchanged | Keep its rows without running the model |
| No face found | Record it, so it isn't re-processed next time |
| Undecodable | Report it in `failed` and retry on the next sync |

`python -m scripts.build_embeddings --full` re-embeds everything. A sync also does a full rebuild automatically if `MODEL_NAME` changes.

## Sample celebrity album

`scripts/download_celebrity_album.py` downloads freely licensed photos from Wikimedia Commons for 10 people: Obama, Biden, Merkel, Macron, Modi, Trudeau, Messi, Cristiano Ronaldo, Shah Rukh Khan and Virat Kohli. It prefers group photos, so other people appear too, and it adds a few distractor photos that contain none of the 10. Every photo is checked with the face model, so it is only kept if a celebrity's face is really in it.

- `data/album/photo_NNN.jpg`: album photos with neutral names, like a real album
- `data/queries/<person>.jpg`: one Wikipedia portrait per person, not in the album; use these to query `/match`
- `data/album_manifest.json`: which celebrities are in each photo, for checking results
- `data/album_credits.csv`: author, licence and source for every image

## Docker

The model pack is baked into the image at build time, so the container starts without downloading anything.

```bash
docker compose up --build
# after adding photos to ./data/album:
curl -X POST localhost:8000/api/v1/album/sync
```

Without compose:

```bash
docker build -t facerecognition .
docker run -p 8000:8000 -v "$PWD/data:/app/data" facerecognition
```

## API

| Method | Path | Description |
|---|---|---|
| GET  | `/api/v1/health` | Model loaded? Photo and face counts |
| GET  | `/api/v1/album` | Indexed photos and the number of faces in each |
| POST | `/api/v1/album/sync` | Incremental sync, same as the script (`?full=true` to re-embed everything) |
| POST | `/api/v1/match` | Upload an image (multipart field `file`) and get every album photo containing that person |
| GET  | `/album-images/{filename}` | Serves an album photo (used for `image_url`) |

`/match` query params: `threshold` (default 0.45), `top_k` (max photos returned, default 20), `all_faces` (default `false`, which searches only the largest face in the upload).

```bash
curl -F "file=@data/queries/barack_obama.jpg" "localhost:8000/api/v1/match"
```

```json
{
  "faces_detected": 1,
  "threshold": 0.45,
  "top_k": 20,
  "results": [{
    "face_index": 0,
    "bbox": {"x1": 120.4, "y1": 88.1, "x2": 301.9, "y2": 320.7},
    "det_score": 0.89,
    "match_count": 2,
    "matches": [
      {"filename": "photo_014.jpg", "score": 0.7731,
       "face_bbox": {"x1": 512.0, "y1": 140.2, "x2": 590.3, "y2": 236.8},
       "image_url": "http://localhost:8000/album-images/photo_014.jpg"},
      {"filename": "photo_003.jpg", "score": 0.6408,
       "face_bbox": {"x1": 88.5, "y1": 201.0, "x2": 160.1, "y2": 290.4},
       "image_url": "http://localhost:8000/album-images/photo_003.jpg"}
    ]
  }]
}
```

`face_bbox` shows where the person is in that album photo. A photo appears only once, scored by its best-matching face.

Errors: `400` for an undecodable image, `413` for a file over `MAX_UPLOAD_MB`, `415` for an unsupported type, `422` when no face is detected, `503` when the album is empty.

## `.npz` format

| Key | Shape | Meaning |
|---|---|---|
| `embeddings` | (F, 512) float32 | One L2-normalised ArcFace embedding per face |
| `face_photo` | (F,) str | Photo each face belongs to |
| `face_bbox` | (F, 4) float32 | Face box in that photo (x1, y1, x2, y2) |
| `face_det_score` | (F,) float32 | Detector confidence |
| `photos` / `photo_hashes` | (P,) str | Every processed photo and its SHA-1, including photos with no faces |
| `model_name`, `built_at` | scalar | Model used and time of the last sync |

## Tuning

- **`MATCH_THRESHOLD`** (cosine similarity): with ArcFace, the same person usually scores **0.5–0.9** and different people score **< 0.3**. Raise it for fewer false positives; lower it if real matches are being missed.
- **Small faces in big group photos**: increase `DET_SIZE` (e.g. 1024). This is slower but finds smaller faces.
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
