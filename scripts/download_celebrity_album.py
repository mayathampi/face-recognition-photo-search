"""Build a sample album of freely-licensed celebrity photos from Wikimedia Commons.

For each person:
  * their Wikipedia lead portrait is saved to data/queries/ (use these to query /match);
  * Commons photos with their name in the title are downloaded as candidates, preferring
    group shots, and each candidate is verified with the face model against the portrait —
    only photos in which a celebrity's face is actually found are kept.
A few group photos containing none of the celebrities are kept as distractors.

Outputs:
  data/album/photo_NNN.jpg      album photos (neutral names, like a real album)
  data/queries/<person>.jpg     one portrait per person
  data/album_manifest.json      which celebrities are in each photo (+ similarity)
  data/album_credits.csv        author / licence / source of every image

Usage:
    python -m scripts.download_celebrity_album
    python -m scripts.download_celebrity_album --per-person 6 --distractors 8
"""

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import requests

from app.config import BASE_DIR, get_settings
from app.ml_module.face_engine import FaceEngine, InvalidImageError, decode_image

logger = logging.getLogger("album")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "FacerecognitionDemo/1.0 (local test-album builder) python-requests"
REQUEST_DELAY = 1.0  # seconds between requests — Wikimedia throttles bursts

# slug -> (Wikipedia article, Commons title search terms)
PEOPLE = {
    "barack_obama": ("Barack Obama", ["Obama Biden", "Obama Merkel", "Obama Modi", "Obama Trudeau", "Barack Obama"]),
    "joe_biden": ("Joe Biden", ["Biden Obama", "Biden Trudeau", "Biden Macron", "Joe Biden"]),
    "angela_merkel": ("Angela Merkel", ["Merkel Macron", "Merkel Obama", "Merkel Modi", "Angela Merkel"]),
    "emmanuel_macron": ("Emmanuel Macron", ["Macron Merkel", "Macron Trudeau", "Macron Modi", "Emmanuel Macron"]),
    "narendra_modi": ("Narendra Modi", ["Modi Obama", "Modi Trudeau", "Modi Macron", "Narendra Modi"]),
    "justin_trudeau": ("Justin Trudeau", ["Trudeau Obama", "Trudeau Macron", "Trudeau Modi", "Justin Trudeau"]),
    "lionel_messi": ("Lionel Messi", ["Messi Argentina", "Messi Barcelona", "Lionel Messi"]),
    "cristiano_ronaldo": ("Cristiano Ronaldo", ["Cristiano Ronaldo Portugal", "Ronaldo Real Madrid", "Cristiano Ronaldo"]),
    "shah_rukh_khan": ("Shah Rukh Khan", ["Shah Rukh Khan with", "Shahrukh Khan", "Shah Rukh Khan"]),
    "virat_kohli": ("Virat Kohli", ["Virat Kohli Anushka", "Kohli India", "Virat Kohli"]),
}
DISTRACTOR_SEARCHES = ["group photo conference", "team photo", "family photo summit", "panel discussion speakers"]

MATCH_SCORE = 0.5  # cosine similarity needed to say a celebrity is in a photo
CANDIDATES_PER_SEARCH = 12


class Wiki:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def _request(self, url: str, **kwargs) -> requests.Response:
        """GET with a polite delay and back-off on 429/5xx (honours Retry-After)."""
        for attempt in range(6):
            time.sleep(REQUEST_DELAY)
            r = self.session.get(url, timeout=60, **kwargs)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** attempt)
                logger.info("HTTP %d — waiting %.0fs", r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

    def _get(self, url: str, **params) -> dict:
        return self._request(url, params=params | {"format": "json", "formatversion": 2}).json()

    def lead_image_title(self, article: str) -> str | None:
        data = self._get(WIKIPEDIA_API, action="query", titles=article, prop="pageimages", piprop="name")
        name = data["query"]["pages"][0].get("pageimage")
        return f"File:{name}" if name else None

    def image_infos(self, *, titles: list[str] | None = None, search: str | None = None,
                    min_width: int = 500) -> list[dict]:
        """Commons file info (thumb URL + licence metadata) by exact titles or by a title search."""
        params = dict(action="query", prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth=1280)
        if titles:
            params["titles"] = "|".join(titles)
        else:
            params |= dict(generator="search", gsrnamespace=6, gsrlimit=CANDIDATES_PER_SEARCH,
                           gsrsearch=" ".join(f'intitle:"{w}"' for w in search.split()) + " filetype:bitmap")
        pages = self._get(COMMONS_API, **params).get("query", {}).get("pages", [])
        infos = []
        for page in sorted(pages, key=lambda p: p.get("index", 0)):
            if not page.get("imageinfo"):
                continue
            ii = page["imageinfo"][0]
            if ii.get("mime") not in ("image/jpeg", "image/png") or ii.get("width", 0) < min_width:
                continue
            meta = ii.get("extmetadata", {})
            infos.append({
                "title": page["title"],
                "thumb_url": ii.get("thumburl") or ii["url"],
                "page_url": ii.get("descriptionurl", ""),
                "artist": _strip_html(meta.get("Artist", {}).get("value", "")),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
            })
        return infos

    def download(self, url: str) -> bytes:
        return self._request(url).content


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--album-dir", type=Path, default=settings.album_dir)
    parser.add_argument("--queries-dir", type=Path, default=BASE_DIR / "data" / "queries")
    parser.add_argument("--per-person", type=int, default=6, help="album photos to collect per celebrity")
    parser.add_argument("--distractors", type=int, default=8, help="photos with none of the celebrities")
    parser.add_argument("--clean", action="store_true", help="empty the album and queries dirs first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("urllib3", "insightface"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.clean:
        for d in (args.album_dir, args.queries_dir):
            shutil.rmtree(d, ignore_errors=True)
    args.album_dir.mkdir(parents=True, exist_ok=True)
    args.queries_dir.mkdir(parents=True, exist_ok=True)

    wiki = Wiki()
    engine = FaceEngine(model_name=settings.model_name, ctx_id=settings.ctx_id, det_size=settings.det_size,
                        min_det_score=settings.min_det_score, model_root=settings.model_root)
    credits: list[dict] = []

    # 1. Reference portraits ------------------------------------------------------
    references: dict[str, np.ndarray] = {}
    used_titles: set[str] = set()
    for slug, (article, _) in PEOPLE.items():
        title = wiki.lead_image_title(article)
        infos = wiki.image_infos(titles=[title], min_width=150) if title else []
        if not infos:
            logger.warning("No usable portrait found for %s (%s) — skipping", article, title)
            continue
        info = infos[0]
        raw = wiki.download(info["thumb_url"])
        faces = engine.detect(decode_image(raw))
        if not faces:
            logger.warning("No face in %s's portrait — skipping", article)
            continue
        references[slug] = faces[0].embedding
        (args.queries_dir / f"{slug}.jpg").write_bytes(raw)
        used_titles.add(info["title"])
        credits.append({"file": f"queries/{slug}.jpg", **info})
        logger.info("Portrait: %s (%s)", article, info["license"])

    slugs = list(references)
    ref_matrix = np.stack([references[s] for s in slugs])

    # 2. Candidates, verified by face ---------------------------------------------
    album: list[dict] = []  # {raw, info, people, face_count}
    per_person = {s: 0 for s in slugs}

    def identify(raw: bytes) -> tuple[dict[str, float], int]:
        faces = engine.detect(decode_image(raw))
        if not faces:
            return {}, 0
        sims = np.stack([f.embedding for f in faces]) @ ref_matrix.T  # (faces, people)
        best = sims.max(axis=0)
        return {s: round(float(best[i]), 3) for i, s in enumerate(slugs) if best[i] >= MATCH_SCORE}, len(faces)

    def consider(info: dict) -> dict | None:
        if info["title"] in used_titles:
            return None
        used_titles.add(info["title"])
        try:
            raw = wiki.download(info["thumb_url"])
            people, face_count = identify(raw)
        except (requests.RequestException, InvalidImageError) as exc:
            logger.warning("Skipping %s: %s", info["title"], exc)
            return None
        return {"raw": raw, "info": info, "people": people, "face_count": face_count}

    for slug in slugs:
        article, searches = PEOPLE[slug]
        for search in searches:
            if per_person[slug] >= args.per_person:
                break
            for info in wiki.image_infos(search=search):
                if per_person[slug] >= args.per_person:
                    break
                item = consider(info)
                if not item or slug not in item["people"]:
                    continue
                album.append(item)
                for s in item["people"]:
                    per_person[s] += 1
                logger.info("  + %-55.55s faces=%d people=%s", info["title"], item["face_count"],
                            ",".join(item["people"]))
        logger.info("%s: %d photo(s)", article, per_person[slug])

    distractors = 0
    for search in DISTRACTOR_SEARCHES:
        for info in wiki.image_infos(search=search):
            if distractors >= args.distractors:
                break
            item = consider(info)
            if item and not item["people"] and item["face_count"] >= 2:
                album.append(item)
                distractors += 1
                logger.info("  + distractor %-45.45s faces=%d", info["title"], item["face_count"])

    # 3. Write album, manifest, credits -------------------------------------------
    rng = np.random.default_rng(7)  # shuffle so photo numbers don't reveal who is in them
    manifest = {}
    for n, i in enumerate(rng.permutation(len(album)), 1):
        item = album[i]
        name = f"photo_{n:03d}.jpg"
        (args.album_dir / name).write_bytes(item["raw"])
        manifest[name] = {"celebrities": item["people"], "face_count": item["face_count"],
                          "source": item["info"]["page_url"]}
        credits.append({"file": f"album/{name}", **item["info"]})

    data_dir = args.album_dir.parent
    (data_dir / "album_manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(data_dir / "album_credits.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "title", "artist", "license", "page_url", "thumb_url"])
        writer.writeheader()
        writer.writerows(credits)

    print(f"\nAlbum: {len(album)} photos, {sum(i['face_count'] for i in album)} faces -> {args.album_dir}")
    print(f"Queries: {len(references)} portraits -> {args.queries_dir}")
    for slug in slugs:
        print(f"  {slug:<20} {per_person[slug]} photo(s)")
    print(f"  {'(distractors)':<20} {distractors} photo(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
