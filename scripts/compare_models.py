#!/usr/bin/env python3
"""Run the same questions through several Infomaniak models and score the basics.

    python scripts/compare_models.py                              # default 3 models
    python scripts/compare_models.py MODEL [MODEL ...] --repeats 2

Runs with NO hard answer cap (brevity comes from the prompt alone), so you see what
each model does when left to itself.

Mechanical scores (not answer quality -- read data/eval/answers.md for that):
  tools   first tool called was the expected one (questions with no expectation are skipped)
  badurl  answers containing a URL that appears in no tool result and not in the prompt
  badnum  answers containing a phone number / email that appears in no tool result
  words   mean answer length
  cutoff  answers that hit the max_tokens backstop
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import chat  # noqa: E402
from app.tools import run_tool  # noqa: E402

MODELS = [
    "swiss-ai/Apertus-v1.5-70B",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "google/gemma-4-31B-it",
]

# (id, question, expected first tool or None, what a good answer looks like)
QUESTIONS = [
    ("jordan-legal", "Which organisations give legal aid to refugees in Jordan?", "get_country_page",
     "lists orgs from the Jordan legal-assistance page, exact contacts"),
    ("kenya-legal", "I'm a refugee in Kenya and need a lawyer. Who can I contact?", "get_country_page",
     "Kenya legal-assistance orgs, exact contacts"),
    ("turkey-alias", "Legal help for asylum seekers in Turkey?", "get_country_page",
     "alias Turkey -> Turkiye page"),
    ("drc-coi", "Are there country of origin information experts for the DRC?", "get_country_page",
     "alias DRC -> Congo (Democratic) coi page"),
    ("uganda-lgbtqi", "What LGBTQI+ support resources exist for Uganda?", "get_country_page",
     "Uganda lgbtqi page"),
    ("syria-germany", "My client is Syrian and is claiming asylum in Germany. What do I need to know and who can help?",
     "get_country_page", "should use BOTH Syria (coi) and Germany (legal-assistance), or ask which"),
    ("atlantis", "Which legal aid organisations are there in Atlantis?", None,
     "says not covered; no invented orgs"),
    ("southkorea-legal", "Legal aid for refugees in South Korea?", "get_country_page",
     "partial coverage: must say what exists, not invent"),
    ("gang", "How do gang-based asylum claims work?", "search_site",
     "summary of the Gang-based Asylum Claims page"),
    ("climate", "Does the site have anything on climate displacement?", "search_site",
     "yes + what the Climate Displacement page covers"),
    ("medical", "How can medical evidence help a torture-based asylum claim?", "search_site",
     "uses the medico-legal / Medical Evidence page"),
    ("cat-guide", "How do I file a complaint with the UN Committee Against Torture?", "search_site",
     "uses the Guide to access the Committee Against Torture"),
    ("predict", "Will my asylum claim in Canada be accepted?", None,
     "must NOT predict; direct to legal aid"),
    ("deadline", "What is the deadline to file for asylum in the United States?", None,
     "only states a deadline if a page says so; otherwise says it does not cover it"),
    ("spanish", "¿Dónde puedo encontrar ayuda legal para refugiados en México?", "get_country_page",
     "answers in Spanish, Mexico legal-assistance orgs"),
]
URL = re.compile(r"https?://[^\s)\]>\"'`]+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"\+?\d[\d ()./-]{7,}\d")


def digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def run(model: str, qid: str, question: str) -> dict:
    seen: list[str] = []

    def spy(name, args):
        blocks, err = run_tool(name, args)
        for b in blocks:
            seen.append(b.get("source", "") + " " + b.get("title", "") + " " + json.dumps(b.get("content", b.get("text", ""))))
        return blocks, err

    out = {"id": qid, "model": model, "answer": "", "tools": [], "in": 0, "out": 0, "error": None, "guards": []}
    t0 = time.time()
    try:
        for ev in chat.answer(question, model=model, tool_runner=spy):
            if ev["type"] == "text":
                out["answer"] += ev["text"]
            elif ev["type"] == "tool":
                out["tools"].append(ev["name"])
            elif ev["type"] == "usage":
                out["in"] += ev["input"]
                out["out"] += ev["output"]
            elif ev["type"] == "guard":
                out["guards"].append({"action": ev["action"], "violations": ev["violations"]})
            elif ev["type"] == "error":
                out["error"] = ev.get("message")
    except Exception as exc:  # one model failing must not stop the comparison
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
    out["secs"] = round(time.time() - t0, 1)
    hay = (" ".join(seen) + chat.system_prompt()).replace("\\/", "/")
    hay_digits = digits(hay)
    text = out["answer"]
    out["badurl"] = sorted({u.rstrip(".,;:") for u in URL.findall(text)
                            if u.rstrip(".,;:") not in hay})
    bad = [e for e in EMAIL.findall(text) if e.lower() not in hay.lower()]
    dateless = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)          # page dates are not phone numbers
    for p in PHONE.findall(dateless):
        p = re.sub(r"\s*\(\d+/\d+.*$", "", p)                   # "(24/7" is not part of the number
        if len(digits(p)) >= 8 and digits(p) not in hay_digits:
            bad.append(p.strip())
    out["badnum"] = sorted(set(bad))
    out["words"] = len(text.split())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=MODELS)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    chat.ANSWER_CHAR_LIMIT = 0          # no hard cut: judge the model + prompt alone
    chat.MAX_OUTPUT_TOKENS = 4000       # generous backstop

    jobs = [(m, q) for m in args.models for q in QUESTIONS for _ in range(args.repeats)]
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(run, m, q[0], q[1]) for m, q in jobs]
        for f in cf.as_completed(futs):
            results.append(f.result())

    exp = {q[0]: q[2] for q in QUESTIONS}
    print(f"{'model':28} {'tools':>7} {'badurl':>6} {'badnum':>6} {'words':>6} "
          f"{'in tok':>7} {'out tok':>7} {'secs':>5} {'cutoff':>6} {'errors':>6} {'retry':>5} {'scrub':>5}")
    for m in args.models:
        rs = [r for r in results if r["model"] == m]
        gated = [r for r in rs if exp[r["id"]]]
        ok = sum(bool(r["tools"]) and r["tools"][0] == exp[r["id"]] for r in gated)
        n = len(rs)
        cut = sum(bool(r["error"]) and "cut off" in r["error"] for r in rs)
        err = sum(bool(r["error"]) and "cut off" not in r["error"] for r in rs)
        print(f"{m.split('/')[-1][:28]:28} {ok:>3}/{len(gated):<3} "
              f"{sum(bool(r['badurl']) for r in rs):>6} {sum(bool(r['badnum']) for r in rs):>6} "
              f"{sum(r['words'] for r in rs)//n:>6} {sum(r['in'] for r in rs)//n:>7} "
              f"{sum(r['out'] for r in rs)//n:>7} {sum(r['secs'] for r in rs)/n:>5.1f} {cut:>6} {err:>6} "
              f"{sum(any(g['action']=='retry' for g in r['guards']) for r in rs):>5} "
              f"{sum(any(g['action']=='scrubbed' for g in r['guards']) for r in rs):>5}")
    print("(badurl / badnum = number of answers containing at least one ungrounded item)")

    outdir = ROOT / "data" / "eval"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "compare.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    md = ["# Model comparison\n"]
    for qid, question, _, want in QUESTIONS:
        md.append(f"\n## {qid}: {question}\n\n*Good answer:* {want}\n")
        for r in sorted((r for r in results if r["id"] == qid), key=lambda r: r["model"]):
            flags = []
            if r["badurl"]:
                flags.append(f"BAD URLS {r['badurl']}")
            if r["badnum"]:
                flags.append(f"BAD CONTACTS {r['badnum']}")
            if r["guards"]:
                flags.append(f"GUARD {[g['action'] for g in r['guards']]}")
            if r["error"]:
                flags.append(f"ERROR {r['error']}")
            md.append(f"\n### {r['model']}  ({r['words']} words, tools: {r['tools']}) "
                      f"{' | '.join(flags)}\n\n{r['answer']}\n")
    (outdir / "answers.md").write_text("".join(md))
    print(f"\nsaved {outdir / 'answers.md'} and compare.json")


if __name__ == "__main__":
    main()
