"""Pull every published page from the WordPress REST API into data/raw/.

Why the REST API and not a crawler: `content.rendered` gives us the page body
without the site nav/footer, plus a real `modified` timestamp per page (this
corpus is full of NGO phone numbers that decay, so the date is part of the
answer). No HTML extraction heuristics needed for boilerplate.

Auth is optional -- all 800+ pages here are published and served publicly.
Set WP_USER + WP_APP_PASSWORD only to also pull drafts/private pages.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import sys

import httpx2 as httpx
from dotenv import load_dotenv

from .extract import clean_title, html_to_text

load_dotenv()

BASE = os.environ.get("WP_BASE_URL", "https://rightsinexile.org").rstrip("/")
POST_TYPES = ["pages", "posts", "interim_measure", "la_portfolio"]
FIELDS = "id,slug,link,title,content,modified,parent,type,status"
RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"


def _auth_header() -> dict[str, str]:
    user, app_pw = os.environ.get("WP_USER"), os.environ.get("WP_APP_PASSWORD")
    if not (user and app_pw):
        return {}
    token = base64.b64encode(f"{user}:{app_pw}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def fetch_type(client: httpx.Client, post_type: str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        r = client.get(
            f"{BASE}/wp-json/wp/v2/{post_type}",
            params={"per_page": 100, "page": page, "_fields": FIELDS, "orderby": "id", "order": "asc"},
        )
        if r.status_code == 404 and page == 1:
            print(f"  {post_type}: not exposed over REST, skipping")
            return []
        # WP returns 400 "rest_post_invalid_page_number" once you walk past the end.
        if r.status_code == 400 and page > 1:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        total_pages = int(r.headers.get("x-wp-totalpages", 1))
        print(f"  {post_type}: page {page}/{total_pages} (+{len(batch)})")
        if page >= total_pages:
            break
        page += 1
    return out


def normalise(rec: dict, post_type: str) -> dict | None:
    body = html_to_text((rec.get("content") or {}).get("rendered", ""))
    if len(body) < 120:  # empty stubs, redirect shells, "page not found"
        return None
    return {
        "id": rec["id"],
        "type": rec.get("type", post_type),
        "slug": rec["slug"],
        "url": rec["link"],
        "title": clean_title((rec.get("title") or {}).get("rendered", "")),
        "parent": rec.get("parent", 0),
        "modified": (rec.get("modified") or "")[:10],
        "text": body,
        "chars": len(body),
    }


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / "pages.jsonl"
    kept, dropped = 0, 0

    headers = {"User-Agent": "H4G-RAG-indexer/1.0 (+rightsinexile.org assistant)", **_auth_header()}
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client, \
            out_path.open("w", encoding="utf-8") as fh:
        for post_type in POST_TYPES:
            print(f"fetching {post_type} ...")
            for rec in fetch_type(client, post_type):
                if rec.get("status") not in (None, "publish"):
                    dropped += 1
                    continue
                doc = normalise(rec, post_type)
                if doc is None:
                    dropped += 1
                    continue
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1

    total_chars = sum(json.loads(l)["chars"] for l in out_path.open(encoding="utf-8"))
    print(f"\nwrote {kept} docs to {out_path} ({dropped} skipped as empty/unpublished)")
    print(f"corpus: {total_chars/1e6:.2f}M chars, roughly {total_chars/3.7/1e3:.0f}k tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
