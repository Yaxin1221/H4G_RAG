# H4G_RAG — Rights in Exile assistant

A grounded chatbot over [rightsinexile.org](https://rightsinexile.org): every answer is
built from that site's pages and carries a link to each page it used.

It is **not** a vector-database RAG pipeline, and that is the central design decision.

## Why there are no embeddings here

Measured against the live site (`make ingest && make index` prints all of this):

| | |
|---|---|
| Published pages | **800** |
| `legal-assistance-by-country` | 233 |
| `country-of-origin-information` | 233 |
| `sexual-orientation-and-gender-identity-country-list` | 234 |
| Everything else (special issues, self-help kits, treaty-body guides, team) | 98 |
| Whole site, boilerplate stripped | **6.8M chars ≈ 1.82M tokens** |

Two facts follow:

**Dumping the whole site into context does not work.** 1.82M tokens against a 1M context
window. Not close enough to fix by trimming.

**Dense-vector retrieval is the wrong tool for 88% of this site.** 700 of the 800 pages
are three *parallel* per-country directories, each following one template: refugee
protection framework → international legal framework → national legal framework → legal
aid organisations with addresses, phone numbers and contact persons. All 233
"X Legal Assistance" pages sit on top of each other in embedding space, so
*"legal help in Jordan"* scores about as well against Lebanon, Kuwait or Israel as
against Jordan, and which one wins is close to arbitrary.

And the URL already encodes `(category, country)` exactly. So retrieval here is a
**lookup, not a similarity search**:

- `data/index.json` — `{category: {country_key: doc_id}}` plus the country catalogue that
  goes in the cached system prompt (~4.9k tokens, well under 1% of the corpus)
- `data/corpus.json` — whole pages, keyed by WordPress id

Claude resolves the country from the question — Burma/Myanmar, the DRC, the Emirates,
Holland, Turkey/Türkiye, "my client is Kurdish" → which country — then calls
`get_country_page(country, category)` and gets the **whole page** back.

**Nothing is chunked, deliberately.** These pages are lists of organisations with contact
details. Splitting them separates an NGO from its phone number, which is not a quality
regression but a wrong answer. The largest page on the site is 88k chars (~24k tokens),
so a whole page fits a tool result comfortably.

For the 98 thematic pages — where a question genuinely needs prose, not a lookup — there
is plain **BM25** over whole pages (`search_site` → `get_page`). Lexical beats semantic
here too: real questions name countries, instruments, organisations and case names, and
those match on the literal token. The index builds in 0.3s with no dependencies.

Citations come from `search_result` content blocks with `citations: {enabled: true}`, so
the model's citations arrive as `search_result_location` objects already carrying the page
URL and title — clickable sources with no lookup table on our side, and no prompting for
a citation format.

## Quickstart

```bash
make setup                  # venv + deps, copies .env.example -> .env
$EDITOR .env                # set ANTHROPIC_API_KEY
make ingest                 # ~10s: pull all 800 pages from the WP REST API
make index                  # build the lookup tables
make check                  # offline checks, no API key needed
make ask Q="Who provides legal aid to refugees in Jordan?"
make serve                  # http://127.0.0.1:8000
```

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
app/corpus.py          lookups, BM25, snippets, catalogue text
app/tools.py           get_country_page / search_site / get_page -> search_result blocks
app/chat.py            system prompt, streaming tool loop, citations
app/server.py          FastAPI + SSE, rate limiting
web/index.html         minimal chat UI, light/dark, mobile
scripts/ask.py         CLI harness
scripts/check.py       offline checks
```

Why the REST API and not a crawler: `content.rendered` excludes the site nav and footer.
Scraping rendered pages instead drags ~1,150 chars of identical chrome into all 800 pages
— about 240k tokens of pure noise, and the single fastest way to wreck ranking. It also
gives a real `modified` date per page, which this corpus needs (see below).

## Keeping it current

`make refresh` re-pulls, rebuilds and re-checks. Run it on a schedule — weekly is fine.

Staleness is the real risk on this site, more than retrieval quality. The pages are full
of NGO phone numbers, email addresses and named contact persons that decay, and parts of
the site have not been touched since 2023. So every page's last-modified date is folded
into the `search_result` title, the system prompt tells the model to surface it whenever
it gives a contact, and the UI repeats the caveat. The date deliberately lives in the
tool result and **not** in the cached system prefix, where a changing value would
invalidate the cache on every request.

## Cost and caching

The system prefix (instructions + 237-country catalogue + thematic page list) is ~4.9k
tokens and carries `cache_control`, so it is a cache read on every request after the
first. Per-question cost is dominated by the pages fetched: a median country page is
~5.3k chars (~1.4k tokens). `make ask` prints `cache_read` / `cache_write` per turn — if
`cache_read` stays 0 across questions, something volatile crept into the prefix.

Effort is set to `medium`: on a website, answer latency matters, and the hard part here
is a lookup rather than a reasoning problem.

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
      keyed by session.

## Status

Everything from the WordPress fetch through to the assembled request is exercised against
the live site by `make check` (30 checks, all passing). The Anthropic call itself has not
yet been run — there were no API credentials on the machine where this was built. Put a
key in `.env` and run `make ask Q="..."` as the first real test.
