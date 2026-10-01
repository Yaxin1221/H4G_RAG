"""Grounding guard: nothing the user can act on may be invented.

The system prompt tells the model to answer only from tool results, but models
break that rule -- fake links, phone numbers from memory, an email with one
letter changed, even a printed "tool call" and "result". Prompts do not stop it,
so the final answer is checked in code against everything the tools returned
this conversation. Anything checkable that appears in no tool result is a
violation: URLs, emails, phone numbers and ISO dates ("page last updated ...").
"""

from __future__ import annotations

import difflib
import re
import unicodedata

URL = re.compile(r"https?://[^\s)\]>\"'`<]+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"\+?\d[\d ()./-]{7,}\d")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
FAKE_TOOL = re.compile(r"\b(?:get_country_page|list_countries|search_site|get_page)\s*\(|"
                       r"\bTool call:|\bResult fetched\b", re.I)

_TRAIL = ".,;:!?*_)]}'\""


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def find_ungrounded(answer: str, grounded_text: str) -> list[tuple[str, str]]:
    """[(kind, item)] for every checkable item in `answer` not present in `grounded_text`."""
    hay = grounded_text
    hay_lower = hay.lower()
    hay_digits = _digits(hay)
    bad: list[tuple[str, str]] = []

    if FAKE_TOOL.search(answer):
        bad.append(("fake-tool", "text that pretends to be a tool call or result"))

    for u in sorted(set(URL.findall(answer))):
        u = u.rstrip(_TRAIL)
        if u not in hay and u.rstrip("/") not in hay:
            bad.append(("url", u))

    for e in sorted(set(EMAIL.findall(answer))):
        e = e.rstrip(_TRAIL)
        if e.lower() not in hay_lower:
            bad.append(("email", e))

    no_dates = ISO_DATE.sub(" ", answer)                      # dates are not phone numbers
    for p in sorted(set(PHONE.findall(no_dates))):
        p = re.sub(r"\s*\(\d+/\d+.*$", "", p).strip()          # "(24/7" is not part of a number
        if re.fullmatch(r"\d{4}\s*[/-]\s*\d{4}", p):               # a year range, not a phone number
            continue
        if len(_digits(p)) >= 8 and _digits(p) not in hay_digits:
            bad.append(("phone", p))

    for d in sorted(set(ISO_DATE.findall(answer))):
        if d not in hay:
            bad.append(("date", d))
    return bad


def _plain(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def repair(answer: str, violations: list[tuple[str, str]], grounded_text: str) -> str:
    """Fix near-miss emails and URLs (an added accent, one changed letter) by swapping in
    the exact value from the tool result. Only when one real value is clearly the
    original: a wrong-but-similar contact is worse than none, so the bar is high."""
    real = {"email": sorted(set(EMAIL.findall(grounded_text))),
            "url": sorted({u.rstrip(_TRAIL) for u in URL.findall(grounded_text)})}
    for kind, item in violations:
        if kind not in real:
            continue
        scored = sorted(((difflib.SequenceMatcher(None, _plain(item), _plain(r)).ratio(), r)
                         for r in real[kind]), reverse=True)
        if scored and scored[0][0] >= 0.9 and (len(scored) == 1 or scored[1][0] < scored[0][0]):
            answer = answer.replace(item, scored[0][1])
    return answer


def scrub(answer: str, violations: list[tuple[str, str]]) -> str:
    """Last resort when a retry still contains ungrounded items: remove them, loudly."""
    for kind, item in violations:
        if kind == "fake-tool":
            continue
        marks = {"url": "[link removed: not found on the site]",
                 "email": "[email removed: not found on the site]",
                 "phone": "[number removed: not found on the site]",
                 "date": "[date removed]"}
        answer = answer.replace(item, marks[kind])
    return answer


def correction(violations: list[tuple[str, str]]) -> str:
    """The hidden message that makes the model rewrite its answer."""
    items = "; ".join(f"{k}: {v}" for k, v in violations[:8])
    return (
        "Your draft cannot be shown: it contains items that appear in NO tool result "
        f"({items}). Rewrite the answer. Use only information from tool results; if you "
        "need a country's or topic's page, call the tool first; never write out a tool call "
        "yourself. If the site does not cover it, say so and point to "
        "https://rightsinexile.org/contact-us/. The user has not seen the draft: do not "
        "mention it, this correction, or any earlier answer; just give the final answer."
    )
