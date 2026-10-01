"""Tool surface. Four tools, no vector search.

`get_country_page` is the load-bearing one: it turns "what legal help exists in
Burma?" into an exact (category, country) lookup and hands back the *whole*
page. Claude does the country resolution (Burma/Myanmar, DRC, the Emirates,
"my client is Kurdish" -> which country), which is precisely the step dense
retrieval gets wrong on a corpus of 234 identically-templated country pages.

Everything a tool returns comes back as a `search_result` content block with
citations enabled, so Claude's answer carries `search_result_location`
citations that already contain the page URL and title -- clickable sources with
no lookup table on our side.
"""

from __future__ import annotations

from typing import Any

from . import corpus

FULL_PAGE_MAX_CHARS = 45_000   # ~11k tokens; longer pages are opened only on request

CATEGORIES = ["legal-assistance", "coi", "lgbtqi"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_country_page",
        "description": (
            "Fetch the full Rights in Exile page for one country in one of the three "
            "per-country directories. Use this for any question about a specific country. "
            "categories: 'legal-assistance' = refugee protection framework plus legal aid "
            "and other assistance organisations; 'coi' = country of origin information "
            "experts and reports; 'lgbtqi' = LGBTQI+ country conditions and resources. "
            "Give the country as the user named it -- aliases like Burma, the DRC, Holland "
            "or Turkey are resolved automatically. Call it once per (country, category) you "
            "need; several calls in parallel is fine."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "Country name, e.g. 'Jordan', 'Burma'."},
                "category": {"type": "string", "enum": CATEGORIES},
            },
            "required": ["country", "category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "list_countries",
        "description": (
            "List every country that has a page in one directory. Only use this when the "
            "user asks which countries a directory covers; for a specific country just "
            "call get_country_page, which says what exists if the page is missing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "enum": CATEGORIES}},
            "required": ["category"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "search_site",
        "description": (
            "Keyword search across the whole site for thematic (non-country) material: "
            "special issues such as gang-based claims or gender in the asylum claim, "
            "self-help kits, guides to UN treaty bodies, medico-legal resources, team "
            "pages. Returns the best-matching page IN FULL plus titles of other matches; if the "
            "best match is not what the user needs, open another with get_page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "scope": {
                    "type": "string",
                    "enum": ["all", "thematic"],
                    "description": "'thematic' excludes the 700 per-country directory pages.",
                },
            },
            "required": ["query", "scope"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_page",
        "description": (
            "Fetch one page in full by the id returned from search_site, or by its URL. "
            "Read a page before citing it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id_or_url": {"type": "string"}},
            "required": ["id_or_url"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _as_search_result(doc: dict) -> dict[str, Any]:
    """A whole page as one citable search_result block.

    `modified` goes in the title because this corpus is full of NGO phone
    numbers and contact persons that decay -- a 2023 address and a 2026 one are
    not equally useful, and the user should see which they got. It stays out of
    the cached system prefix, where a changing value would break the cache.
    """
    return {
        "type": "search_result",
        "source": doc["url"],
        "title": f"{doc['title']} (rightsinexile.org, page last updated {doc['modified']})",
        "content": [{"type": "text", "text": doc["text"]}],
        "citations": {"enabled": True},
    }


def _text(msg: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": msg}]


def _country_miss(category: str, country: str) -> str:
    """Tell the model exactly what exists instead of letting it guess."""
    cov = corpus.country_coverage(country)
    if cov:
        name, cats = cov
        have = ", ".join(f"'{c}'" for c in cats) or "none"
        return (
            f"{name} has no '{category}' page. The directories that do cover {name}: {have}. "
            f"Do not guess: tell the user '{category}' is not covered for {name}"
            + (" and offer those instead." if cats else ".")
        )
    near = corpus.suggest_countries(country)
    hint = f" Did the user mean: {', '.join(near)}? Ask if unclear." if near else ""
    return (
        f"No country matching {country!r} exists in any directory.{hint} Do not guess: "
        f"if it is not a spelling or alias issue, tell the user this country is not covered "
        f"and offer a thematic search instead."
    )


def run_tool(name: str, args: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Execute a tool call. Returns (tool_result content blocks, is_error)."""
    try:
        if name == "get_country_page":
            category, country = args["category"], args["country"]
            doc = corpus.lookup_country(category, country)
            if doc is None:
                return _text(_country_miss(category, country)), False
            return [_as_search_result(doc)], False

        if name == "list_countries":
            names = corpus.countries_with(args["category"])
            return _text(f"{len(names)} countries in '{args['category']}': " + "; ".join(names)), False

        if name == "search_site":
            query, scope = args["query"], args.get("scope", "all")
            index, all_docs = corpus.load()
            only = None
            if scope == "thematic":
                only = {p["id"] for p in index["prose"]}
            hits = corpus.bm25().search(query, k=6, only=only)
            if not hits:
                return _text("No matches. Try different keywords, or say you could not find it."), False
            top_id, _ = hits[0]
            top = all_docs[top_id]
            rest = hits[1:] if top["chars"] <= FULL_PAGE_MAX_CHARS else hits
            lines = [
                f"[{doc_id}] {all_docs[doc_id]['title']} -- {all_docs[doc_id]['url']} "
                f"(updated {all_docs[doc_id]['modified']})"
                for doc_id, _ in rest
            ]
            listing = ("Other matches (call get_page with an id to open one in full):\n"
                       + "\n".join(lines)) if lines else ""
            if top["chars"] <= FULL_PAGE_MAX_CHARS:
                return ([_as_search_result(top)] + (_text(listing) if listing else [])), False
            # Very long page: do not dump it unasked; let the model choose.
            return _text("Ranked matches, none opened yet (call get_page with an id to read "
                         "one in full before answering):\n" + listing), False

        if name == "get_page":
            doc = corpus.get_doc(str(args["id_or_url"]))
            if doc is None:
                return _text(f"No page found for {args['id_or_url']!r}."), False
            return [_as_search_result(doc)], False

        return _text(f"Unknown tool {name!r}."), True
    except Exception as exc:  # a tool crash must not kill the turn
        return _text(f"Tool {name} failed: {type(exc).__name__}: {exc}"), True
