"""Loads the built index and serves lookups. No vector database, by design."""

from __future__ import annotations

import difflib
import functools
import json
import math
import pathlib
import re
from collections import Counter

from ingest.countries import candidates

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "a an the and or of in on for to from with is are was were be been by at as "
    "this that these those it its i you we they what which who how do does can "
    "there their about into if not no any all more most other some such than then".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


@functools.lru_cache(maxsize=1)
def load() -> tuple[dict, dict]:
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    corpus = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
    if not corpus:
        raise RuntimeError("empty corpus -- run `make ingest && make index`")
    return index, corpus


class Bm25:
    """Plain BM25 over whole pages.

    Lexical, not semantic, and that is the right call for this corpus: real
    questions name countries, instruments, organisations and case names, and
    those match on the literal token. It also never confuses the 234
    near-identical country templates with each other the way cosine similarity
    on dense vectors does.
    """

    k1, b = 1.5, 0.75

    def __init__(self, docs: dict[str, dict]):
        self.ids = list(docs)
        self.tf = {i: Counter(_tokens(docs[i]["title"] + "\n" + docs[i]["text"])) for i in self.ids}
        self.len = {i: sum(self.tf[i].values()) or 1 for i in self.ids}
        self.avg = sum(self.len.values()) / len(self.ids)
        df = Counter(t for i in self.ids for t in self.tf[i])
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 5, only: set[str] | None = None) -> list[tuple[str, float]]:
        q = _tokens(query)
        scored: list[tuple[str, float]] = []
        for i in self.ids:
            if only is not None and i not in only:
                continue
            tf, dl = self.tf[i], self.len[i]
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


@functools.lru_cache(maxsize=1)
def bm25() -> Bm25:
    _, corpus = load()
    return Bm25(corpus)


def lookup_country(category: str, country: str) -> dict | None:
    """(category, country) -> whole document, or None."""
    index, corpus = load()
    table = index["directory"].get(category)
    if not table:
        return None
    for key in candidates(country):
        doc_id = table.get(key)
        if doc_id:
            return corpus[doc_id]
    # last resort: a unique substring match against indexed keys
    for key in candidates(country):
        hits = [v for k, v in table.items() if key and (key in k or k in key)]
        if len(hits) == 1:
            return corpus[hits[0]]
    return None


def get_doc(doc_id_or_url: str) -> dict | None:
    _, corpus = load()
    if doc_id_or_url in corpus:
        return corpus[doc_id_or_url]
    needle = doc_id_or_url.rstrip("/").rsplit("/", 1)[-1]
    for doc in corpus.values():
        if doc["slug"] == needle or doc["url"].rstrip("/") == doc_id_or_url.rstrip("/"):
            return doc
    return None


def snippet(doc: dict, query: str, width: int = 320) -> str:
    q = set(_tokens(query))
    best, best_hits = doc["text"][:width], -1
    for start in range(0, min(len(doc["text"]), 20_000), width // 2):
        window = doc["text"][start:start + width]
        hits = sum(1 for t in _tokens(window) if t in q)
        if hits > best_hits:
            best, best_hits = window, hits
    return " ".join(best.split())


_CATEGORY_FLAGS = (("L", "legal-assistance"), ("C", "coi"), ("G", "lgbtqi"))


def countries_with(category: str) -> list[str]:
    """Names of all countries that have a page in `category`."""
    index, _ = load()
    flag = next(f for f, c in _CATEGORY_FLAGS if c == category)
    return [r["country"] for r in index["catalogue"] if flag in r["have"]]


def country_coverage(country: str) -> tuple[str, list[str]] | None:
    """(display name, categories that exist) for a country, or None if unknown."""
    index, _ = load()
    by_key = {r["key"]: r for r in index["catalogue"]}
    for key in candidates(country):
        row = by_key.get(key)
        if row:
            return row["country"], [c for f, c in _CATEGORY_FLAGS if f in row["have"]]
    return None


def suggest_countries(country: str, n: int = 4) -> list[str]:
    """Closest country names, for when a lookup misses (typos, odd spellings)."""
    index, _ = load()
    by_key = {r["key"]: r["country"] for r in index["catalogue"]}
    found: list[str] = []
    for key in candidates(country):
        for m in difflib.get_close_matches(key, by_key, n=n, cutoff=0.6):
            if by_key[m] not in found:
                found.append(by_key[m])
    return found[:n]


def prose_catalogue_text(limit_chars: int = 12_000) -> str:
    index, _ = load()
    out, total = [], 0
    for p in index["prose"]:
        line = f"- {p['title']} ({p['url']}) [updated {p['modified']}]"
        if total + len(line) > limit_chars:
            break
        out.append(line)
        total += len(line)
    return "\n".join(out)
