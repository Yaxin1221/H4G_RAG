"""FastAPI endpoint. The API key stays server-side; the browser only sees SSE.

Run:  make serve      ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import pathlib
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .chat import answer
from .corpus import load

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parents[1]
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_HOUR", "30"))

@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    index, corpus_docs = load()
    print(f"corpus loaded: {len(corpus_docs)} pages, "
          f"{len(index['catalogue'])} countries, {len(index['prose'])} thematic pages")
    yield


app = FastAPI(title="Rights in Exile assistant", lifespan=lifespan)

# A public chatbot is an open wallet: cap it before anything else.
# Fine for one process; move to Redis the moment you run more than one.
_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)


def _rate_limit(request: Request) -> None:
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "unknown"))
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT} questions per hour.")
    q.append(now)


@app.get("/api/health")
def health() -> dict:
    index, corpus_docs = load()
    return {"ok": True, "pages": len(corpus_docs), "countries": len(index["catalogue"])}


@app.post("/api/chat")
async def chat(request: Request) -> StreamingResponse:
    _rate_limit(request)
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if len(question) > 2000:
        raise HTTPException(400, "question too long")
    history = body.get("history") or []

    def sse():
        try:
            for ev in answer(question, history):
                if ev["type"] == "usage":          # server-side telemetry only
                    print(f"usage {ev}", flush=True)
                    continue
                if ev["type"] == "done":
                    yield f"data: {json.dumps({'type': 'done', 'history': _serialisable(ev['history'])})}\n\n"
                    continue
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as exc:
            print(f"chat failed: {type(exc).__name__}: {exc}", flush=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong.'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _serialisable(history: list) -> list:
    """SDK content blocks -> plain JSON so the browser can hand history back."""
    out = []
    for msg in history:
        content = msg["content"]
        if isinstance(content, list):
            content = [c if isinstance(c, dict) else c.model_dump(exclude_none=True) for c in content]
        out.append({"role": msg["role"], "content": content})
    return out


@app.get("/")
def index_html() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")
