"""Turn data/raw/pages.jsonl into the two structures the bot actually queries.

The whole point of this design: 703 of the 800 pages are three *parallel*
per-country directories (legal assistance / COI / LGBTQI+), each following the
same template. Dense-vector retrieval is bad at exactly this shape -- all 234
"X Legal Assistance" pages sit on top of each other in embedding space, so
"legal help in Jordan" scores about as well against Lebanon or Kuwait. And the
URL already encodes (category, country) exactly.

So retrieval here is a *lookup*, not a similarity search:

  data/index.json   {category: {country_key: doc_id}} + the country catalogue
                    that goes in the cached system prompt
  data/corpus.json  {doc_id: full document}  -- whole pages, never chunked

Chunking is deliberately absent. These pages are lists of organisations with
addresses, phone numbers and contact persons; splitting them separates an NGO
from its phone number, which is not a quality regression but a wrong answer.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

from .countries import normalise

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pages.jsonl"

# WordPress parent page id -> our category slug
DIRECTORY_PARENTS = {
    7793: "legal-assistance",
    8030: "coi",
    8104: "lgbtqi",
}
CATEGORY_LABELS = {
    "legal-assistance": "legal assistance and refugee protection framework, by country",
    "coi": "country of origin information (COI) experts and reports, by country",
    "lgbtqi": "LGBTQI+ resources and country conditions, by country",
}
# The site's own title suffixes. "LBGTQI" is a typo on one live page (Iraq).
SUFFIX = re.compile(
    r"^(?P<country>.*?)\s*(?:Legal Assistance|COI|Country of Origin Information"
    r"|L[GB]{2}TQI\+?\s*Resources)\s*$",
    re.IGNORECASE,
)


def main() -> int:
    if not RAW.exists():
        print(f"missing {RAW} -- run `make ingest` first", file=sys.stderr)
        return 1

    docs = [json.loads(line) for line in RAW.open(encoding="utf-8")]
    corpus: dict[str, dict] = {}
    directory: dict[str, dict[str, str]] = {c: {} for c in DIRECTORY_PARENTS.values()}
    display: dict[str, str] = {}   # country_key -> the site's spelling
    prose: list[dict] = []
    duplicates: list[tuple] = []

    for d in docs:
        doc_id = str(d["id"])
        corpus[doc_id] = d
        category = DIRECTORY_PARENTS.get(d["parent"])
        m = SUFFIX.match(d["title"]) if category else None
        if category and m and m.group("country").strip():
            country = m.group("country").strip()
            key = normalise(country)
            d["country"] = country
            d["category"] = category
            prev = directory[category].get(key)
            if prev is not None:
                # The site has a few true duplicates (Isle of Man, DPRK) from
                # re-created pages. Keep the fuller, more recently edited one
                # rather than whichever happened to come last in the feed.
                keep = max((prev, doc_id), key=lambda i: (corpus[i]["chars"], corpus[i]["modified"]))
                duplicates.append((category, country, prev, doc_id, keep))
                directory[category][key] = keep
            else:
                directory[category][key] = doc_id
            display.setdefault(key, country)
        else:
            # Section landing pages, guides, special issues, team bios -- and the
            # handful of directory children that aren't country pages at all.
            d["category"] = "prose"
            prose.append(d)

    # Catalogue for the cached system prompt: one row per country, with which of
    # the three directories exist for it.
    catalogue = []
    for key in sorted(display, key=lambda k: display[k]):
        have = "".join(
            flag if key in directory[cat] else "-"
            for cat, flag in (("legal-assistance", "L"), ("coi", "C"), ("lgbtqi", "G"))
        )
        catalogue.append({"country": display[key], "key": key, "have": have})

    index = {
        "site": "https://rightsinexile.org",
        "categories": CATEGORY_LABELS,
        "directory": directory,
        "catalogue": catalogue,
        "prose": [
            {"id": str(d["id"]), "title": d["title"], "url": d["url"],
             "modified": d["modified"], "chars": d["chars"]}
            for d in sorted(prose, key=lambda x: -x["chars"])
        ],
    }

    (ROOT / "data" / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "data" / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    dir_chars = sum(corpus[i]["chars"] for cat in directory.values() for i in cat.values())
    prose_chars = sum(d["chars"] for d in prose)
    print(f"countries indexed : {len(catalogue)}")
    for cat in directory:
        print(f"  {cat:<18} {len(directory[cat])} pages")
    print(f"prose pages       : {len(prose)}")
    for cat, country, a, b, keep in duplicates:
        print(f"  duplicate on site: {cat}/{country} -> ids {a},{b}; kept {keep}")
    print(f"directory text    : {dir_chars/1e6:.2f}M chars (~{dir_chars/3.7/1e3:.0f}k tokens)")
    print(f"prose text        : {prose_chars/1e6:.2f}M chars (~{prose_chars/3.7/1e3:.0f}k tokens)")
    print(f"whole site        : ~{(dir_chars+prose_chars)/3.7/1e6:.2f}M tokens "
          f"-- does NOT fit a 1M context window, hence the lookup design")
    return 0


if __name__ == "__main__":
    sys.exit(main())
