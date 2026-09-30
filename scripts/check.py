#!/usr/bin/env python3
"""Offline checks -- everything up to the Anthropic call. No API key needed."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import corpus                      # noqa: E402
from app.chat import system_blocks          # noqa: E402
from app.tools import TOOLS, run_tool       # noqa: E402

fails: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(label)


print("corpus")
index, docs = corpus.load()
check("pages loaded", len(docs) > 750, f"{len(docs)} pages")
check("countries indexed", len(index["catalogue"]) > 200, f"{len(index['catalogue'])} countries")
check("thematic pages", 50 < len(index["prose"]) < 200, f"{len(index['prose'])} pages")
for cat in ("legal-assistance", "coi", "lgbtqi"):
    check(f"directory {cat}", len(index["directory"][cat]) > 200,
          f"{len(index['directory'][cat])} pages")

print("\ncountry aliases (the step dense retrieval gets wrong)")
for given, expect in [
    ("Burma", "Myanmar"), ("the DRC", "Congo"), ("Turkey", "rkiye"),
    ("Ivory Coast", "Ivoire"), ("Holland", "Netherlands"), ("USA", "United States"),
    ("Czechia", "Czech"), ("Jordan", "Jordan"),
]:
    doc = corpus.lookup_country("legal-assistance", given)
    check(f"{given} -> {expect}", doc is not None and expect.lower() in doc["title"].lower(),
          doc["title"] if doc else "not found")

print("\nunknown country must not silently resolve")
doc = corpus.lookup_country("legal-assistance", "Wakanda")
check("Wakanda -> None", doc is None, doc["title"] if doc else "")
out, err = run_tool("get_country_page", {"country": "Wakanda", "category": "coi"})
check("tool says 'no page' rather than guessing",
      not err and out[0]["type"] == "text" and "No 'coi' page exists" in out[0]["text"])

print("\ntool results are citable search_result blocks")
out, err = run_tool("get_country_page", {"country": "Jordan", "category": "legal-assistance"})
blk = out[0]
check("no error", not err)
check("type is search_result", blk.get("type") == "search_result")
check("source is the live URL", str(blk.get("source", "")).startswith("https://rightsinexile.org/"))
check("citations enabled", blk.get("citations") == {"enabled": True})
check("title carries the page date", "last updated 20" in blk.get("title", ""))
check("content is text blocks", isinstance(blk.get("content"), list)
      and blk["content"][0]["type"] == "text" and len(blk["content"][0]["text"]) > 500)
check("page is whole, not chunked", "Tel:" in blk["content"][0]["text"]
      or "Email" in blk["content"][0]["text"])

print("\nthematic search")
out, err = run_tool("search_site", {"query": "gang-based asylum claims", "scope": "thematic"})
check("returns ranked list", not err and "Gang-based" in out[0]["text"])
hits = corpus.bm25().search("guide to the Committee Against Torture", 5)
titles = [docs[i]["title"] for i, _ in hits]
check("treaty-body guide ranks top-3",
      any("Committee Against Torture" in t for t in titles[:3]), "; ".join(titles[:3]))
out, err = run_tool("get_page", {"id_or_url": hits[0][0]})
check("get_page returns search_result", not err and out[0]["type"] == "search_result")

print("\nprompt shape")
blocks = system_blocks()
chars = sum(len(b["text"]) for b in blocks)
check("single cached system block", len(blocks) == 1
      and blocks[0]["cache_control"] == {"type": "ephemeral"})
check("prefix is small enough to be cheap", chars < 60_000, f"{chars} chars (~{chars//4} tokens)")
check("prefix has no volatile content", "2026-" not in blocks[0]["text"].split("## Thematic")[0])
check("tools declared", {t["name"] for t in TOOLS}
      == {"get_country_page", "search_site", "get_page"})
check("tools are strict", all(t.get("strict") for t in TOOLS))

print()
if fails:
    print(f"{len(fails)} check(s) failed: {', '.join(fails)}")
    sys.exit(1)
print("all checks passed")
