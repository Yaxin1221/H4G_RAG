"""Tool surface. Three tools, no vector search.

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
        "name": "search_site",
        "description": (
            "Keyword search across the whole site for thematic (non-country) material: "
            "special issues such as gang-based claims or gender in the asylum claim, "
            "self-help kits, guides to UN treaty bodies, medico-legal resources, team "
            "pages. Returns ranked titles with snippets -- follow up with get_page to read "
            "one in full before answering from it."
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


def run_tool(name: str, args: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Execute a tool call. Returns (tool_result content blocks, is_error)."""
    try:
        if name == "get_country_page":
            category, country = args["category"], args["country"]
            doc = corpus.lookup_country(category, country)
            if doc is None:
                index, _ = corpus.load()
                return _text(
                    f"No '{category}' page exists for {country!r}. Do not guess: tell the "
                    f"user this country is not covered in that directory and offer the "
                    f"other two directories or a thematic search instead."
                ), False
            return [_as_search_result(doc)], False

        if name == "search_site":
            query, scope = args["query"], args.get("scope", "all")
            index, all_docs = corpus.load()
            only = None
            if scope == "thematic":
                only = {p["id"] for p in index["prose"]}
            hits = corpus.bm25().search(query, k=6, only=only)
            if not hits:
                return _text("No matches. Try different keywords, or say you could not find it."), False
            lines = [
                f"[{doc_id}] {all_docs[doc_id]['title']} -- {all_docs[doc_id]['url']} "
                f"(updated {all_docs[doc_id]['modified']}, score {score:.1f})\n"
                f"    {corpus.snippet(all_docs[doc_id], query)}"
                for doc_id, score in hits
            ]
            return _text(
                "Ranked matches (snippets only -- call get_page with an id to read one in "
                "full before citing it):\n\n" + "\n\n".join(lines)
            ), False

        if name == "get_page":
            doc = corpus.get_doc(str(args["id_or_url"]))
            if doc is None:
                return _text(f"No page found for {args['id_or_url']!r}."), False
            return [_as_search_result(doc)], False

        return _text(f"Unknown tool {name!r}."), True
    except Exception as exc:  # a tool crash must not kill the turn
        return _text(f"Tool {name} failed: {type(exc).__name__}: {exc}"), True
