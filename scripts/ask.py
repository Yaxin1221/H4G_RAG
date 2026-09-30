#!/usr/bin/env python3
"""CLI harness -- test answers without running the server.

    python scripts/ask.py "who gives legal aid to refugees in Jordan?"
    python scripts/ask.py            # interactive
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.chat import answer  # noqa: E402

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def ask(question: str, history: list | None = None) -> list:
    cites: list[dict] = []
    out_history = history or []
    for ev in answer(question, history):
        kind = ev["type"]
        if kind == "text":
            print(ev["text"], end="", flush=True)
        elif kind == "cite":
            cites.append(ev)
        elif kind == "tool":
            print(f"{DIM}[{ev['name']} {ev['args']}]{RESET}", file=sys.stderr, flush=True)
        elif kind == "usage":
            print(f"{DIM}[in {ev['input']} out {ev['output']} "
                  f"cache_read {ev['cache_read']} cache_write {ev['cache_write']}]{RESET}",
                  file=sys.stderr, flush=True)
        elif kind == "error":
            print(f"\n{BOLD}error:{RESET} {ev.get('message')} {ev.get('details','')}")
        elif kind == "done":
            out_history = ev["history"]
    if cites:
        print(f"\n\n{BOLD}Sources{RESET}")
        seen = set()
        for c in cites:
            if c["url"] in seen:
                continue
            seen.add(c["url"])
            print(f"  - {c['title']}\n    {c['url']}")
    print()
    return out_history


def main() -> int:
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return 0
    history: list = []
    while True:
        try:
            q = input(f"{BOLD}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not q:
            continue
        if q in {"exit", "quit"}:
            return 0
        history = ask(q, history)


if __name__ == "__main__":
    raise SystemExit(main())
