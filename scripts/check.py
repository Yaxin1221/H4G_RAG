#!/usr/bin/env python3
"""Offline checks -- everything up to the model call. No API key needed."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import corpus                      # noqa: E402
from app import chat                        # noqa: E402
from app.chat import system_prompt          # noqa: E402
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
      not err and out[0]["type"] == "text" and "Do not guess" in out[0]["text"])
out, _ = run_tool("get_country_page", {"country": "Wakanda", "category": "coi"})
check("miss on unknown country has no suggestion of a bogus match", "Did the user mean" not in out[0]["text"]
      or "Wakanda" in out[0]["text"])
out, _ = run_tool("get_country_page", {"country": "Jordon", "category": "coi"})
check("typo gets a suggestion", "Jordan" in out[0]["text"], out[0]["text"][:90])
partial = [r for r in index["catalogue"] if r["have"] != "LCG"][0]
missing = next(c for f, c in (("L", "legal-assistance"), ("C", "coi"), ("G", "lgbtqi")) if f not in partial["have"])
out, _ = run_tool("get_country_page", {"country": partial["country"], "category": missing})
check("partial coverage names what exists", "directories that do cover" in out[0]["text"], partial["country"])
out, err = run_tool("list_countries", {"category": "lgbtqi"})
check("list_countries works", not err and "Jordan" in out[0]["text"])

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
out2, _ = run_tool("search_site", {"query": "climate displacement", "scope": "thematic"})
check("search_site returns best page in full", out2[0]["type"] == "search_result"
      and "Climate" in out2[0]["title"])
check("get_page returns search_result", not err and out[0]["type"] == "search_result")

print("\nprompt shape")
prompt = system_prompt()
chars = len(prompt)
check("prefix is small enough to be cheap", chars < 60_000, f"{chars} chars (~{chars//4} tokens)")
check("prefix has no volatile content", "2026-" not in prompt.split("## Thematic")[0])
check("country list is not in the prompt", "Afghanistan [" not in prompt)
check("thematic list is not in the prompt", "rightsinexile.org/special-issues/" not in prompt)
check("prompt is lean", chars < 12_000, f"{chars} chars")

print("\ngrounding guard")
from app import guard                                                   # noqa: E402
page = run_tool("get_country_page", {"country": "Jordan", "category": "legal-assistance"})[0][0]
hay = page["source"] + " " + page["title"] + " " + page["content"][0]["text"] + " https://rightsinexile.org/contact-us/"
real_email = next(iter(guard.EMAIL.findall(page["content"][0]["text"])))
real_phone = "+962 6 462 4009"
good = f"See {page['source']} (page last updated 2024-01-26). Email {real_email}, tel {real_phone}."
check("grounded answer passes", guard.find_ungrounded(good, hay) == [], str(guard.find_ungrounded(good, hay)))
for label, text, kind in [
    ("invented URL", "See https://rightsinexile.org/country/canada/legal-assistance/ now.", "url"),
    ("mutated email", "Write to " + real_email.replace("a", "e", 1) + " today.", "email"),
    ("invented phone", "Call +1 416 555 0199 for help.", "phone"),
    ("invented date", "Page last updated 2031-02-03.", "date"),
    ("fake tool call", "Tool call:\nget_country_page(country='DRC')", "fake-tool"),
]:
    kinds = {k for k, _ in guard.find_ungrounded(text, hay)}
    check(f"guard catches {label}", kind in kinds, str(kinds))
check("year range is not a phone number", guard.find_ungrounded("UNHCR 2016/2018 guidelines", hay) == [])
fixed = guard.repair("Mail " + real_email.replace("a", "\u00e1", 1) + " now", [("email", real_email.replace("a", "\u00e1", 1))], hay)
check("accented email is repaired to the exact one", real_email in fixed, fixed)
fixed = guard.repair("Mail zzzz@example.org now", [("email", "zzzz@example.org")], hay)
check("dissimilar email is not 'repaired'", "zzzz@example.org" in fixed)
check("scrub removes the invented URL",
      "canada" not in guard.scrub("see https://rightsinexile.org/country/canada/legal-assistance/ x",
                                  [("url", "https://rightsinexile.org/country/canada/legal-assistance/")]))

print("\nanswer length limit")
chat.ANSWER_CHAR_LIMIT, chat.OVERRUN_CHARS = 100, 200
piece, cut = chat._clip("", "short")
check("under the limit passes through", piece == "short" and not cut)
body = "\n".join(f"- item {i} " + "x" * 20 for i in range(20))
shown, cut = "", False
for i in range(0, len(body), 7):          # stream in small pieces
    p, cut = chat._clip(shown, body[i:i + 7])
    shown += p
    if cut:
        break
check("long answer is cut", cut and len(shown) < len(body))
check("cut lands on a line end", shown.endswith(("x", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
      and len(shown) <= 100 + 60, repr(shown[-25:]))
prose = "This is one sentence. " * 3 + "And a very long second sentence that keeps going and going. Third."
shown, cut = "", False
for i in range(0, len(prose), 5):
    p, cut = chat._clip(shown, prose[i:i + 5])
    shown += p
    if cut:
        break
check("prose is cut at a sentence end", cut and shown.endswith("."), repr(shown[-30:]))
chat.ANSWER_CHAR_LIMIT, chat.OVERRUN_CHARS = 2000, 600

check("tools declared", {t["name"] for t in TOOLS}
      == {"get_country_page", "list_countries", "search_site", "get_page"})
check("tools are strict", all(t.get("strict") for t in TOOLS))

print()
if fails:
    print(f"{len(fails)} check(s) failed: {', '.join(fails)}")
    sys.exit(1)
print("all checks passed")
