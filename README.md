# H4G_RAG — Rights in Exile assistant

A grounded chatbot over [rightsinexile.org](https://rightsinexile.org): every answer is
built from that site's pages and carries a link to each page it used.

The model is [Gemma](https://huggingface.co/google) (`google/gemma-4-31B-it`) served
by Infomaniak AI Tools: Swiss-hosted, OpenAI-compatible API. Any other model Infomaniak
offers can be swapped in with `INFOMANIAK_MODEL`.

It is **not** a vector-database RAG pipeline, and that is the central design decision.

## Why there are no embeddings here

Measured against the live site (`make ingest && make index` prints all of this):

| | |
|---|---|
| Published pages | **800** |
| `legal-assistance-by-country` | 233 countries |
| `country-of-origin-information` | 233 countries |
| `sexual-orientation-and-gender-identity-country-list` | 234 countries |
| Duplicate country pages on the site (Isle of Man, DPRK), one of each pair kept in the lookup | 2 |
| Everything else (special issues, self-help kits, treaty-body guides, team) | 98 |
| Whole site, boilerplate stripped | **6.8M chars ≈ 1.82M tokens** |

Two facts follow:

**Dumping the whole site into context does not work.** 1.82M tokens is far beyond the
context window of any model this runs on. Not close enough to fix by trimming.

**Dense-vector retrieval is the wrong tool for 88% of this site.** 700 of the 800 pages
are three *parallel* per-country directories, each following one template: refugee
protection framework → international legal framework → national legal framework → legal
aid organisations with addresses, phone numbers and contact persons. All 233
"X Legal Assistance" pages sit on top of each other in embedding space, so
*"legal help in Jordan"* scores about as well against Lebanon, Kuwait or Israel as
against Jordan, and which one wins is close to arbitrary.

And the URL already encodes `(category, country)` exactly. So retrieval here is a
**lookup, not a similarity search**:

- `data/index.json` — `{category: {country_key: doc_id}}` plus the country catalogue
- `data/corpus.json` — whole pages, keyed by WordPress id

The model passes the country as the user wrote it and the lookup resolves it in code
(`ingest/countries.py`): Burma/Myanmar, the DRC, the Emirates, Holland, Turkey/Türkiye.
`get_country_page(country, category)` then returns the **whole page**. A miss never leaves
the model guessing: if the country exists but that directory does not, the tool names the
directories that do; if nothing matches, it suggests close spellings ("Jordon" → Jordan)
or says the country is not covered.

**Nothing is chunked, deliberately.** These pages are lists of organisations with contact
details. Splitting them separates an NGO from its phone number, which is not a quality
regression but a wrong answer. The largest page on the site is 88k chars (~24k tokens).

For the 98 thematic pages — where a question genuinely needs prose, not a lookup — there
is plain **BM25** over whole pages. Lexical beats semantic here too: real questions name
countries, instruments, organisations and case names, and those match on the literal
token. The index builds in 0.3s with no dependencies. `search_site` returns the best match
**in full** plus the titles of the other matches, so most thematic questions take one tool
call; only a best match over 45k chars is left unopened for the model to choose.

The four tools:

| Tool | What it does |
|---|---|
| `get_country_page` | exact `(country, category)` lookup, whole page back |
| `list_countries` | every country in one directory, for "which countries do you cover?" |
| `search_site` | BM25 over whole pages; best match in full, other titles listed |
| `get_page` | open any page by id or URL |

The country catalogue and the thematic page list are **not** in the system prompt. The
prompt is instructions only (~4.7k chars, ~1.2k tokens); what exists is discovered through
the tools.

## Sources and the grounding guard

Every tool result starts with `SOURCE:` (the page URL) and `TITLE:` (which includes the
last-modified date). The model links the pages it used inline as Markdown, and the UI
lists every page fetched during the turn under "Sources".

The prompt says to answer only from tool results. Models break that rule: invented links,
a phone number from memory, an email with one letter changed, even a printed fake "tool
call". So the final answer is checked in code (`app/guard.py`) before the user sees it.
Every URL, email address, phone number and ISO date in the answer must appear in a tool
result from this conversation (or in the system prompt). If one does not:

1. **Repair** — a near-miss email or URL (an added accent, one changed letter) is replaced
   by the exact value from the page, only when a single real value is clearly the original.
2. **Retry** — otherwise the model gets one hidden correction and rewrites the answer.
3. **Scrub** — if the rewrite still fails, the item is replaced by a visible marker such as
   `[number removed: not found on the site]`. An answer that fakes a tool call is replaced
   by a fixed "could not verify" message.

The cost: the answer is buffered for checking, so it arrives in one piece instead of
streaming token by token. The UI shows "looking up …" while tools run.

The guard only checks things that can be matched literally. An invented organisation
name or a wrong legal claim in plain prose passes it; that still rests on the prompt and
on human testing (see below).

Answer length is also enforced in code, because "be brief" is not reliably obeyed:
`ANSWER_CHAR_LIMIT` (default 2000) cuts at the next sentence or line end and appends a
note, and `MAX_OUTPUT_TOKENS` (default 1500) is the backstop. Both can be set in `.env`.

## How a question is answered

The split between country-specific and other questions is not made in code: the model
picks the tool, steered by the system prompt (`INSTRUCTIONS` in `app/chat.py`). Everything
after that choice (lookup, search, grounding check) is deterministic.

```
                              ┌──────────────────┐
                              │  User question   │
                              └────────┬─────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │ server.py: 30 questions/hour per  │
                     │ IP, non-empty, max 2000 chars     │
                     └─────────────────┬─────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │ chat.py: system prompt + last 10  │
                     │ user turns + question + 4 tools   │
                     └─────────────────┬─────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
        ┌───────────▶│  MODEL decides what it needs      │◀───────────────┐
        │            │  (max 6 tool rounds)              │                │
        │            └──┬──────────────┬──────────────┬──┘                │
        │               │              │              │                   │
        │   COUNTRY-SPECIFIC     NOT COUNTRY-SPECIFIC │ no tool needed    │
        │               │              │              │ (or origin vs     │
        │               ▼              ▼              │ asylum country    │
        │   ┌──────────────────┐ ┌──────────────────┐ │ unclear: ask      │
        │   │ get_country_page │ │ search_site      │ │ the user)         │
        │   │ country +        │ │ scope "thematic" │ │                   │
        │   │ category:        │ │ BM25 keywords    │ │                   │
        │   │ legal-assistance │ │ over ~98 pages,  │ │                   │
        │   │ | coi | lgbtqi   │ │ top 6 hits       │ │                   │
        │   └────────┬─────────┘ └────────┬─────────┘ │                   │
        │            ▼                    ▼           │                   │
        │   alias / spelling      ┌── any hits? ──┐   │                   │
        │   resolved, exact       │ no            │ yes                   │
        │   table lookup          ▼               ▼   │                   │
        │    │         │      "No matches"   top page │                   │
        │   hit       miss    (reword once   ≤ 45k    │                   │
        │    │         │      or say not     chars?   │                   │
        │    ▼         ▼      found)         │    │   │                   │
        │  whole    country known,          yes   no  │                   │
        │  page     category missing:        │    │   │                   │
        │  + URL    "these directories       ▼    ▼   │                   │
        │  + last   exist instead"        top page  titles only,          │
        │  updated  country unknown:      in full   model opens one       │
        │           close names, or       + other   with get_page         │
        │           "not covered"         titles                          │
        │               │                    │                │           │
        └───────────────┴── tool results added to the conversation ───────┘
                                                      │
                             model writes text, no more tool calls
                                                      ▼
                     ┌───────────────────────────────────┐
                     │ Draft, cut at ~2000 chars on a    │
                     │ sentence end                      │
                     └─────────────────┬─────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │ guard.py: every URL, email, phone │
                     │ number, ISO date must appear in a │
                     │ tool result; no fake "tool call"  │
                     └──────┬─────────────────────┬──────┘
                        all grounded          violations
                            │                     ▼
                            │        repair near-miss email/URL
                            │        (≥ 0.9 match to a real one)
                            │                     │
                            │              still violations?
                            │               │            │
                            │           1st time     2nd time
                            │               │            │
                            │        hidden retry    scrub: "[link removed…]"
                            │        with correction or fallback message if
                            │        (back to MODEL) it faked a tool call
                            ▼                            │
                     ┌───────────────────────────────────┴┐
                     │ Answer + source links shown;       │
                     │ history returned to the browser    │
                     └────────────────────────────────────┘
```

Not drawn, to keep the chart readable:

- **Mixed questions** (e.g. "gender-based claims in Jordan") take both branches in the
  same turn: the thematic page for the framework, the country page for who can help.
- **"Which countries do you cover?"** uses `list_countries`, which returns the names for
  one directory.
- **Hard stops**: a provider content filter, hitting the output token limit, or more than
  6 tool rounds each end the turn with an error instead of an answer.

## Quickstart

You need `git`, `make` and Python 3 (developed on 3.14; 3.10 or newer should work), plus the
project's Infomaniak token (see step 2). macOS and Linux work as is; on Windows use WSL.

**1. Clone and install**

```bash
git clone https://github.com/Yaxin1221/H4G_RAG.git
cd H4G_RAG
make setup                  # creates .venv, installs deps, copies .env.example -> .env
```

**2. Add your Infomaniak credentials**

For now everyone uses one shared token. Ask the repo owner
([@Yaxin1221](https://github.com/Yaxin1221)) for the API token and the product id; they
are sent privately and are not in the repo. Put both in `.env`:

```bash
INFOMANIAK_API_KEY=the-token-you-were-sent
INFOMANIAK_PRODUCT_ID=the-product-id-you-were-sent
# INFOMANIAK_MODEL=google/gemma-4-31B-it    # optional, this is the default
```

`.env` is git-ignored. Keep the token out of commits, issues and chat channels: usage on
it is billed to one account.

If you have your own Infomaniak account instead: create an API token with scope
`ai-tools` at [manager.infomaniak.com](https://manager.infomaniak.com), then look up the
product id with it:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.infomaniak.com/1/ai
```

**3. Fetch the site and build the lookup**

The fetched pages are not in the repo, so this step is required after cloning.

```bash
make ingest                 # ~10s: pull all 800 pages from the WP REST API
make index                  # build data/index.json and data/corpus.json
make check                  # offline checks, no API key needed; should end "all checks passed"
```

**4. Ask something**

```bash
make ask Q="Who provides legal aid to refugees in Jordan?"    # one question from the CLI
make serve                                                    # web UI at http://127.0.0.1:8000
```

`make help` lists every target. To update later: `git pull`, then `make setup` if
`requirements.txt` changed, then `make refresh` to re-pull the site.

The WordPress REST API serves every published page **without authentication**, so no
WordPress credentials are needed. `WP_USER` / `WP_APP_PASSWORD` in `.env` exist only if
you later want drafts or private pages — and note that a wp-admin login password does not
work for REST auth; you need an Application Password (wp-admin → Users → Profile →
Application Passwords).

## Layout

```
ingest/fetch_wp.py     pull published pages from /wp-json/wp/v2/*  -> data/raw/pages.jsonl
ingest/extract.py      Elementor HTML -> text, hrefs folded into markdown links
ingest/countries.py    country-name normalisation + alias table
ingest/build_index.py  classify by WP parent id -> data/index.json + data/corpus.json
app/corpus.py          lookups, BM25, country coverage and spelling suggestions
app/tools.py           get_country_page / list_countries / search_site / get_page
app/chat.py            system prompt, tool loop, length limits, guard retry
app/guard.py           grounding check: find, repair, scrub ungrounded items
app/server.py          FastAPI + SSE, rate limiting
web/index.html         minimal chat UI, Markdown rendering, light/dark, mobile
scripts/ask.py         CLI harness
scripts/check.py       offline checks
scripts/compare_models.py   same questions through several models, mechanical scores
```

Why the REST API and not a crawler: `content.rendered` excludes the site nav and footer.
Scraping rendered pages instead drags ~1,150 chars of identical chrome into all 800 pages
— about 240k tokens of pure noise, and the single fastest way to wreck ranking. It also
gives a real `modified` date per page, which this corpus needs (see below).

## Keeping it current

`make refresh` re-pulls, rebuilds and re-checks. Run it on a schedule — weekly is fine.

Staleness is the real risk on this site, more than retrieval quality. The pages are full
of NGO phone numbers, email addresses and named contact persons that decay, and 446 of
the 800 pages were last modified in 2023. So every page's last-modified date is folded
into the `TITLE:` line of the tool result, the system prompt tells the model to surface it
whenever it gives a contact, and the UI repeats the caveat. Because the date comes from a
tool result, the guard also rejects a "last updated" date the model made up.

## Cost

The system prompt is static and small (~1.2k tokens), so per-question cost is dominated
by the pages fetched: a median country page is ~5.4k chars (~1.5k tokens). In the model
comparison below a question used 9.6k to 11.1k input tokens on average, depending on the
model, in total across its tool rounds.
A guard retry adds one more model call. `make ask` prints input/output tokens per call,
and `cache_read` when the provider reports cached prompt tokens.

## Comparing models

```bash
.venv/bin/python scripts/compare_models.py                    # Apertus, Qwen, Gemma
.venv/bin/python scripts/compare_models.py MODEL [MODEL ...] --repeats 2
```

Runs 15 fixed questions (country lookups, aliases, partial coverage, a country that does
not exist, thematic questions, a prediction request, a deadline question, one in Spanish)
through each model with the hard answer cap off. It prints mechanical scores per model
(right first tool, ungrounded URLs and contacts, answer length, tokens, latency, guard
retries and scrubs) and writes every answer to `data/eval/answers.md`. The scores do not
measure answer quality; read the answers for that. `data/eval/` is git-ignored.

## Before this goes public

- [ ] **Rate limiting is per-process.** `RATE_LIMIT_PER_HOUR` (default 30/IP) works for
      one uvicorn worker. Move it to Redis before running more than one.
- [ ] **Have someone at AsyLex read `INSTRUCTIONS` in `app/chat.py`.** It is the whole
      safety boundary: answer only from sources, never state a threshold or deadline that
      wasn't read, no case-specific advice, "the site doesn't cover that" is a valid
      answer, direct case questions to a legal aid organisation.
- [ ] **Test the refusal paths deliberately.** Ask things the site cannot answer and
      confirm it declines instead of inventing an organisation or a deadline. That is the
      failure mode that actually matters for this audience.
- [ ] **The bot has no tools with side effects** and must not get any. Retrieved page
      content is data, not instructions — that is stated in the system prompt, but the
      real defence is that there is nothing for an injected instruction to do.
- [ ] Put it behind TLS and a real origin; `X-Accel-Buffering: no` is already set so nginx
      won't buffer the SSE stream.
- [ ] Decide whether conversation history stays client-side (as now) or moves server-side
      keyed by session. Client-held history is filtered to user/assistant/tool turns, so
      it cannot carry a system prompt, but a client can still send forged tool results,
      and the guard treats those as grounded.

## Status

`make check` exercises everything up to the model call against the live site, plus the
guard and the length limit: 52 checks, all passing, no API key needed. The model call
itself has been run through `scripts/compare_models.py` (15 questions × 2 repeats × 3
models). In that run the guard intervened on 6 of 90 answers: invented or altered URLs,
phone numbers, an email address, a page date and one fake tool call. By model that is
3 of 30 for Apertus, 3 of 30 for Gemma and 0 of 30 for Qwen; one Qwen answer (the Spanish
question) was cut off at the 4000-token backstop instead. Thirty answers per model is too
few to rank them on. The project started on Apertus and now runs on Gemma; Apertus stays
in the comparison list. Answer quality has not been reviewed by anyone at AsyLex yet.
