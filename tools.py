"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
HTTP_TIMEOUT = 30.0
SUMMARY_CHARS = 600
FETCH_CHARS = 12000
ARXIV_MIN_GAP = 3.0                                      # seconds between two arXiv calls (their usage policy)
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}
USER_AGENT = "deep-research-lab/1.0 (student project)"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    The wait is the server's Retry-After when it sent one, else base * 2**attempt plus a random jitter (so many
    clients do not come back at the same instant). Both are capped at `cap`. The last failed attempt re-raises
    without sleeping; any other exception is not retried and propagates at once.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(max(float(exc.retry_after), 0.0), cap)
            else:
                backoff = base * 2 ** attempt
                delay = min(backoff + random.uniform(0, backoff * 0.25), cap)
            time.sleep(delay)


def _retry_after(response):
    """Seconds from a Retry-After header (numeric form), else None."""
    try:
        return max(float(response.headers.get("Retry-After", "")), 0.0)
    except (TypeError, ValueError):
        return None


def _request(method, url, **kwargs):
    """One HTTP call. Transient failures (429/5xx, network errors) become RetryableError; other 4xx raise HTTPStatusError."""
    try:
        response = httpx.request(method, url, timeout=HTTP_TIMEOUT, follow_redirects=True,
                                 headers={"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code} from {url.split('?')[0]}", _retry_after(response))
    response.raise_for_status()
    return response


def _redact(text):
    """Remove the Exa key (and any exaApiKey=... query parameter) from text that may reach the agent."""
    text = re.sub(r"(exaApiKey=)[^&\s'\")]+", r"\1[REDACTED]", str(text))
    key = (os.getenv("EXA_API_KEY") or "").strip()
    return text.replace(key, "[REDACTED]") if key else text


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {_redact(exc)}"[:500]


def _clip(value, limit=SUMMARY_CHARS):
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _clamp(value, low, high, default):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


# ---- TODO 2: arXiv ----
_arxiv_lock = threading.Lock()
_arxiv_last_call = [None]   # monotonic time of the last arXiv request, shared by all threads (subagents run in parallel)
_OPERATORS = {"and", "or", "not", "andnot"}


def _arxiv_terms(query):
    words = re.findall(r"[^\W_]+(?:-[^\W_]+)*", str(query or ""))
    return [w for w in words if w.lower() not in _OPERATORS]


def _arxiv_get(params):
    with _arxiv_lock:   # serialises the callers, so the gap holds even with several researchers in parallel
        last = _arxiv_last_call[0]
        if last is not None and time.monotonic() - last < ARXIV_MIN_GAP:
            time.sleep(ARXIV_MIN_GAP - (time.monotonic() - last))
        try:
            return _request("GET", ARXIV_URL, params=params)
        finally:
            _arxiv_last_call[0] = time.monotonic()


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}.
    Use a few plain keywords (e.g. "world model latent dynamics"), not a sentence. Good for primary research papers."""
    try:
        terms = _arxiv_terms(query)
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{t}" for t in terms), "sortBy": "submittedDate",
                  "sortOrder": "descending", "start": 0, "max_results": _clamp(max_results, 1, 30, 10)}
        response = with_retry(lambda: _arxiv_get(params), attempts=6, base=3.0, cap=60.0)
        records = []
        for entry in ET.fromstring(response.text).findall("a:entry", ARXIV_NS):
            entry_id = (entry.findtext("a:id", "", ARXIV_NS) or "").strip()
            if "/abs/" not in entry_id:   # arXiv reports a malformed query as an <entry> whose id is an error URL
                continue
            paper_id = re.sub(r"v\d+$", "", entry_id.split("/abs/", 1)[1])
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": (entry.findtext("a:published", "", ARXIV_NS) or "")[:10],
                "title": _clip(entry.findtext("a:title", "", ARXIV_NS), 300),
                "summary": _clip(entry.findtext("a:summary", "", ARXIV_NS)),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001 - tools never raise
        return _error(exc)


# ---- TODO 3: Hugging Face ----
def _hf_records(items, use_ai_summary=False):
    records = []
    for item in items if isinstance(items, list) else []:
        paper = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
        if not isinstance(paper, dict) or not paper.get("id"):
            continue
        summary = (paper.get("ai_summary") if use_ai_summary else None) or paper.get("summary") or item.get("summary")
        records.append({
            "id": paper["id"],
            "url": f"https://huggingface.co/papers/{paper['id']}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _clip(paper.get("title") or item.get("title"), 300),
            "summary": _clip(summary),
            "upvotes": paper.get("upvotes") or 0,
            "github": paper.get("githubRepo") or "",
            "stars": paper.get("githubStars") or 0,
        })
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        limit = _clamp(limit, 1, 100, 30)
        keyword = " ".join(str(keyword or "").lower().split())
        params = {"limit": 100 if keyword else limit}   # a keyword filter needs a wide window to find anything
        if date:
            params["date"] = date
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        records = _hf_records(response.json())
        if keyword:
            words = keyword.split()
            records = [r for r in records if all(w in f"{r['title']} {r['summary']}".lower() for w in words)]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records[:limit], ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}. Use a short topic phrase."""
    try:
        query = " ".join(str(query or "").split())
        if not query:
            return "NO RESULTS"
        params = {"q": query, "limit": _clamp(limit, 1, 50, 10)}
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params))
        records = _hf_records(response.json(), use_ai_summary=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
# The free Exa endpoint answers a rate-limited call with HTTP 200 and a short notice as the "page text", flagging it in
# result._meta. We do not trust one exact flag name: any truthy _meta entry whose name mentions rate/limit/throttl
# counts, and so does a short text that talks about a rate limit.
_RATE_TEXT = re.compile(r"rate[ -]?limit|too many requests|exaApiKey", re.IGNORECASE)


def _meta_flags_rate_limit(meta):
    if isinstance(meta, dict):
        return any((re.search(r"rate|limit|throttl", str(k), re.IGNORECASE) and bool(v)) or _meta_flags_rate_limit(v)
                   for k, v in meta.items())
    return False


def _sse_json(body):
    """The JSON-RPC message of a server-sent-events (or plain JSON) body."""
    messages = []
    for line in body.splitlines():
        if line.startswith("data:"):
            try:
                messages.append(json.loads(line[5:].strip()))
            except ValueError:
                continue
    if not messages:
        messages = [json.loads(body)]
    return next((m for m in messages if isinstance(m, dict) and ("result" in m or "error" in m)), messages[0])


def _exa_call(name, arguments):
    """Call one Exa MCP tool over plain HTTP JSON-RPC; return its text. Rate limits are retried; other errors raise."""
    key = (os.getenv("EXA_API_KEY") or "").strip()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def attempt():
        response = _request("POST", EXA_URL, json=payload, headers=headers, params={"exaApiKey": key} if key else None)
        try:
            message = _sse_json(response.text)
        except ValueError as exc:
            raise RuntimeError("Exa answered with something that is not JSON-RPC") from exc
        if "error" in message:
            error = message["error"] if isinstance(message["error"], dict) else {"message": message["error"]}
            if _RATE_TEXT.search(str(error.get("message", ""))) or error.get("code") == 429:
                raise RetryableError(f"Exa rate limit: {error.get('message')}")
            raise RuntimeError(f"Exa error {error.get('code')}: {error.get('message')}")
        result = message.get("result") or {}
        text = "\n".join(p.get("text", "") for p in result.get("content", []) if p.get("type") == "text").strip()
        if _meta_flags_rate_limit(result.get("_meta")) or (len(text) < 1500 and _RATE_TEXT.search(text)):
            raise RetryableError("Exa free-tier rate limit (HTTP 200 with a rate-limit notice)")
        if result.get("isError"):
            raise RuntimeError(f"Exa tool error: {text[:300]}")
        return text

    try:
        return with_retry(attempt, attempts=6, base=3.0, cap=60.0)
    except RetryableError as exc:
        hint = "" if key else " (no EXA_API_KEY set: the free tier is rate limited, get a key at dashboard.exa.ai)"
        raise RuntimeError(f"still failing after retries: {exc}{hint}") from exc


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        query = " ".join(str(query or "").split())
        if not query:
            return "NO RESULTS"
        arguments = {"query": query, "objective": objective.strip() or f"Find pages that answer: {query}",
                     "numResults": _clamp(num_results, 1, 10, 5)}
        text = _exa_call("web_search_exa", arguments)
        return text[:FETCH_CHARS] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        url = str(url or "").strip()
        if not re.match(r"https?://", url):
            return "ERROR: ValueError: url must start with http:// or https://"
        text = _exa_call("web_fetch_exa", {"urls": [url]})
        return text[:FETCH_CHARS] if text else "NO RESULTS"
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
