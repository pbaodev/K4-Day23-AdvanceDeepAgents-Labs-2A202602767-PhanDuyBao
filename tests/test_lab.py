"""Offline tests (no network, no LLM, no sandbox):  pip install pytest && pytest -q"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import check_citations  # noqa: E402
import finalize_citations  # noqa: E402
import research  # noqa: E402
import tools  # noqa: E402


# ---------- with_retry ----------
def test_retry_succeeds_after_failures(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise tools.RetryableError("x")
        return "ok"

    assert tools.with_retry(flaky, base=1.0, cap=30.0) == "ok"
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 1.25 and 2.0 <= sleeps[1] <= 2.5      # exponential + jitter


def test_retry_honours_retry_after_and_cap(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    errors = iter([tools.RetryableError("x", retry_after=7), tools.RetryableError("x", retry_after=500)])

    def fn():
        try:
            raise next(errors)
        except StopIteration:
            return "done"

    assert tools.with_retry(fn, cap=30.0) == "done"
    assert sleeps == [7, 30.0]


def test_retry_last_attempt_does_not_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    def always():
        raise tools.RetryableError("boom")

    with pytest.raises(tools.RetryableError):
        tools.with_retry(always, attempts=3)
    assert len(sleeps) == 2


def test_retry_does_not_retry_other_errors(monkeypatch):
    monkeypatch.setattr(tools.time, "sleep", lambda s: pytest.fail("must not sleep"))
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("bug")

    with pytest.raises(ValueError):
        tools.with_retry(bad)
    assert len(calls) == 1


# ---------- tools (httpx mocked) ----------
def fake_response(status=200, text="", json_data=None, headers=None):
    request = httpx.Request("GET", "https://example.org")
    if json_data is not None:
        return httpx.Response(status, json=json_data, headers=headers, request=request)
    return httpx.Response(status, text=text, headers=headers, request=request)


ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-02T10:00:00Z</published>
<title>World
   Models  Survey</title><summary>  A long
 summary. </summary></entry>
<entry><id>http://arxiv.org/api/errors#bad</id><title>Error</title><summary>bad</summary></entry></feed>"""


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    tools._arxiv_last_call[0] = None


def test_arxiv_parses_and_normalises(monkeypatch):
    seen = {}

    def fake(method, url, **kw):
        seen.update(url=url, params=kw["params"])
        return fake_response(text=ATOM)

    monkeypatch.setattr(tools.httpx, "request", fake)
    out = json.loads(tools.arxiv_search.invoke({"query": 'world: "model" AND', "max_results": 99}))
    assert out == [{"id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "published": "2025-01-02",
                    "title": "World Models Survey", "summary": "A long summary."}]
    assert seen["url"].startswith("https://")
    assert seen["params"]["search_query"] == "all:world AND all:model"
    assert seen["params"]["max_results"] == 30


def test_arxiv_empty_query_no_network(monkeypatch):
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: pytest.fail("no network call expected"))
    assert tools.arxiv_search.invoke({"query": ' "": '}) == "NO RESULTS"


def test_arxiv_keeps_three_seconds_between_calls(monkeypatch):
    sleeps, clock = [], [100.0]
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: fake_response(text=ATOM))
    tools.arxiv_search.invoke({"query": "a"})
    clock[0] += 1.0
    tools.arxiv_search.invoke({"query": "b"})
    assert sleeps == [pytest.approx(2.0)]


def test_arxiv_retries_429_then_gives_error(monkeypatch):
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return fake_response(429, headers={"Retry-After": "2"})

    monkeypatch.setattr(tools.httpx, "request", fake)
    out = tools.arxiv_search.invoke({"query": "world model"})
    assert out.startswith("ERROR:") and "429" in out
    assert len(calls) == 6


def test_tool_never_raises_on_transport_error(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(tools.httpx, "request", boom)
    assert tools.hf_search_papers.invoke({"query": "x"}).startswith("ERROR:")


HF_ITEMS = [
    {"paper": {"id": "2501.1", "title": "World model A", "summary": "about worlds", "upvotes": 5, "githubRepo": "g/a",
               "githubStars": 9, "publishedAt": "2025-01-03T00:00:00Z"}},
    {"paper": {"id": "2501.2", "title": "Other", "summary": "a world model too", "upvotes": 50, "ai_summary": "short",
               "publishedAt": "2025-01-04T00:00:00Z"}},
    {"paper": {"title": "no id"}},
    {"paper": {"id": "2501.3", "title": "Unrelated", "summary": "nothing", "upvotes": 99}},
]


def test_hf_daily_filters_and_sorts(monkeypatch):
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: fake_response(json_data=HF_ITEMS))
    out = json.loads(tools.hf_daily_papers.invoke({"keyword": "World Model"}))
    assert [r["id"] for r in out] == ["2501.2", "2501.1"]
    assert out[1] == {"id": "2501.1", "url": "https://huggingface.co/papers/2501.1", "published": "2025-01-03",
                      "title": "World model A", "summary": "about worlds", "upvotes": 5, "github": "g/a", "stars": 9}


def test_hf_daily_no_match(monkeypatch):
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: fake_response(json_data=HF_ITEMS))
    assert tools.hf_daily_papers.invoke({"keyword": "quantum"}) == "NO RESULTS"


def test_hf_search_prefers_ai_summary(monkeypatch):
    items = [{"paper": {"id": "1", "title": "t", "summary": "long", "ai_summary": "short"}}]
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: fake_response(json_data=items))
    assert json.loads(tools.hf_search_papers.invoke({"query": "x"}))[0]["summary"] == "short"


def sse(result=None, error=None):
    message = {"jsonrpc": "2.0", "id": 1, **({"result": result} if result is not None else {"error": error})}
    return fake_response(text=f"event: message\ndata: {json.dumps(message)}\n\n")


def test_exa_rate_limit_on_http_200_is_retried(monkeypatch):
    limited = sse({"content": [{"type": "text", "text": "You've hit the Exa free MCP rate limit. Use exaApiKey."}],
                   "_meta": {"rateLimited": True}})
    ok = sse({"content": [{"type": "text", "text": "Result page text"}]})
    answers = iter([limited, limited, ok])
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: next(answers))
    assert tools.web_search.invoke({"query": "world models"}) == "Result page text"


def test_exa_meta_flag_alone_triggers_retry(monkeypatch):
    long_text = "something that does not mention it " * 5
    answers = iter([sse({"content": [{"type": "text", "text": long_text}], "_meta": {"x-ratelimit-hit": "1"}}),
                    sse({"content": [{"type": "text", "text": "fine"}]})])
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: next(answers))
    assert tools.web_fetch.invoke({"url": "https://example.org"}) == "fine"


def test_exa_key_is_never_returned(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123456")

    def fake(method, url, **kw):
        assert kw["params"] == {"exaApiKey": "secret-key-123456"}
        raise httpx.ConnectError("failed for https://mcp.exa.ai/mcp?exaApiKey=secret-key-123456")

    monkeypatch.setattr(tools.httpx, "request", fake)
    out = tools.web_search.invoke({"query": "x"})
    assert out.startswith("ERROR:") and "secret-key-123456" not in out


def test_exa_rpc_error_and_arguments(monkeypatch):
    sent = {}

    def fake(method, url, **kw):
        sent.update(kw["json"]["params"])
        return sse(error={"code": -32602, "message": "bad arguments"})

    monkeypatch.setattr(tools.httpx, "request", fake)
    out = tools.web_fetch.invoke({"url": "https://example.org/p"})
    assert out.startswith("ERROR:") and "bad arguments" in out
    assert sent == {"name": "web_fetch_exa", "arguments": {"urls": ["https://example.org/p"]}}


def test_web_fetch_truncates(monkeypatch):
    monkeypatch.setattr(tools.httpx, "request", lambda *a, **k: sse({"content": [{"type": "text", "text": "x" * 50000}]}))
    assert len(tools.web_fetch.invoke({"url": "https://example.org"})) == 12000


def test_web_search_builds_objective(monkeypatch):
    sent = {}

    def fake(method, url, **kw):
        sent.update(kw["json"]["params"]["arguments"])
        return sse({"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setattr(tools.httpx, "request", fake)
    tools.web_search.invoke({"query": "world models", "num_results": 3})
    assert sent["objective"] and sent["numResults"] == 3 and sent["query"] == "world models"


# ---------- check_citations ----------
SOURCES = [{"n": 1, "id": "a", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-01", "source": "arxiv"},
           {"n": 2, "id": "b", "url": "https://example.org/b", "title": "B", "date": "n.d.", "source": "web"}]
GOOD = """# T

## TL;DR
- one [1][2]

## References
[1] A. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-01)
[2] B. web. https://example.org/b (n.d.)
"""


def test_check_ok():
    assert check_citations.check(GOOD, SOURCES) == []


@pytest.mark.parametrize("mutate,expected", [
    (lambda r: r.replace("[2]", "[3]", 1), "[3] cited but missing"),
    (lambda r: r.replace("- one [1][2]", "- one [1]"), "source [2] never cited"),
    (lambda r: r.replace("## References", "## Refs"), "References"),
    (lambda r: r.replace("[2] B. web.", "[2] B. web. https://other.org and"), "exactly one URL"),
    (lambda r: r.replace("https://example.org/b (n.d.)", "https://example.org/zzz (n.d.)"), "differs"),
    (lambda r: r.replace("[2] B. web. https://example.org/b (n.d.)\n", ""), "no line"),
    (lambda r: r + "[2] B again. https://example.org/b\n", "2 lines for [2]"),
    (lambda r: r + "[9] Ghost. https://example.org/g\n", "not in sources"),
])
def test_check_detects(mutate, expected):
    problems = check_citations.check(mutate(GOOD), SOURCES)
    assert any(expected in p for p in problems), problems


def test_check_bundled_references_line():
    bundled = GOOD.replace("[2] B. web. https://example.org/b (n.d.)", "[2] B; C; D. https://example.org/b; https://x.org/c")
    assert check_citations.check(bundled, SOURCES)


def test_check_sources_rules():
    assert check_citations.check(GOOD, []) == ["no sources in sources.json"]
    dup = SOURCES + [dict(SOURCES[1], n=3)]
    assert any("more than one source" in p for p in check_citations.check(GOOD, dup))
    bad = [dict(SOURCES[0], n="1"), dict(SOURCES[1], url="ftp://x")]
    assert len(check_citations.check(GOOD, bad)) >= 2


def test_check_group_citations_and_ignored_patterns():
    body = GOOD.replace("- one [1][2]", "- one [1, 2]")
    assert check_citations.check(body, SOURCES) == []
    assert check_citations.check(GOOD.replace("- one [1][2]", "- one [1-2]"), SOURCES) == []
    ignored = GOOD.replace("- one [1][2]", "- one [1][2] `[7]` [8](https://x.org) and a fence:\n```\n[9]\n```")
    assert check_citations.check(ignored, SOURCES) == []


def test_check_reference_numbers_do_not_count_as_citations():
    only_refs = GOOD.replace("- one [1][2]", "- nothing cited")
    problems = check_citations.check(only_refs, SOURCES)
    assert "source [1] never cited" in problems and "source [2] never cited" in problems


def test_check_agrees_with_finalizer():
    body = "# T\n\n## TL;DR\n- uses [2, 1] and [5-6]\n"
    sources = SOURCES + [{"n": 5, "url": "https://e.org/5", "title": "Five", "source": "web"},
                         {"n": 6, "url": "https://e.org/6", "title": "Six", "source": "hf-search", "date": "2025-02-02"},
                         {"n": 7, "url": "https://e.org/7", "title": "Unused", "source": "web"}]
    report, new_sources, problems = finalize_citations.finalize(body, sources)
    assert problems == [] and len(new_sources) == 4
    assert check_citations.check(report, new_sources) == []


def test_check_main_exit_codes(tmp_path):
    report, sources = tmp_path / "r.md", tmp_path / "s.json"
    report.write_text(GOOD)
    sources.write_text(json.dumps(SOURCES))
    assert check_citations.main(["x", str(report), str(sources)]) == 0
    report.write_text(GOOD.replace("[2]", "[3]", 1))
    assert check_citations.main(["x", str(report), str(sources)]) == 1
    assert check_citations.main(["x", str(tmp_path / "missing"), str(sources)]) == 1


# ---------- research.py ----------
def test_slugify():
    assert research.slugify("survey about world model") == "survey-about-world-model"
    assert research.slugify("../../x") == "x"
    assert research.slugify("  !!!  ") == "topic" and research.slugify("") == "topic"
    long = research.slugify("a" * 100 + " b")
    assert len(long) <= 60 and not long.endswith("-")


def test_summarize_counts_messages_and_dicts():
    messages = [
        SimpleNamespace(tool_calls=[{"name": "task"}, {"name": "task"}, {"name": "write_todos"}],
                        usage_metadata={"input_tokens": 10, "output_tokens": 5}),
        {"tool_calls": [{"name": "task"}], "usage_metadata": {"input_tokens": 1, "output_tokens": 2}},
        SimpleNamespace(tool_calls=[], usage_metadata=None),
        SimpleNamespace(content="plain"),
    ]
    assert research.summarize(messages, 12.34, "m") == {
        "model": "m", "elapsed_s": 12.3, "subagent_calls": 3, "tool_calls": {"task": 3, "write_todos": 1},
        "tokens": {"input": 11, "output": 7}}


class FakeBackend:
    def __init__(self, files):
        self.files = files

    def download_files(self, paths):
        return [SimpleNamespace(path=p, content=self.files.get(p), error=None) for p in paths]


def test_save_outputs_writes_three_files(tmp_path):
    files = {research.REPORT_PATH: GOOD.encode(), research.SOURCES_PATH: json.dumps(SOURCES).encode()}
    path = research.save_outputs(FakeBackend(files), "Survey About X", [], 3.0, "m", reports_dir=tmp_path / "reports")
    assert path == tmp_path / "reports" / "survey-about-x.md" and path.read_text() == GOOD
    meta = json.loads((tmp_path / "reports" / "survey-about-x.meta.json").read_text())
    assert meta["topic"] == "Survey About X" and meta["n_sources"] == 2 and meta["source_families"] == ["arxiv", "web"]
    assert json.loads((tmp_path / "reports" / "survey-about-x.sources.json").read_text()) == SOURCES


@pytest.mark.parametrize("files", [
    {},
    {research.REPORT_PATH: b"  \n", research.SOURCES_PATH: b"[]"},
    {research.REPORT_PATH: GOOD.encode()},
    {research.REPORT_PATH: GOOD.encode(), research.SOURCES_PATH: b"{not json"},
    {research.REPORT_PATH: GOOD.encode(), research.SOURCES_PATH: b'{"a": 1}'},
])
def test_save_outputs_writes_nothing_on_failure(tmp_path, files):
    with pytest.raises(RuntimeError):
        research.save_outputs(FakeBackend(files), "t", [], 1.0, "m", reports_dir=tmp_path / "reports")
    assert not (tmp_path / "reports").exists() or not list((tmp_path / "reports").iterdir())


def test_main_without_topic_returns_2():
    assert research.main("   ") == 2


def test_main_failure_returns_1_and_cleans_up(monkeypatch, capsys):
    events = []

    class Box:
        def execute(self, cmd):
            events.append("execute")

    class Ctx:
        def __enter__(self):
            return Box()

        def __exit__(self, *exc):
            events.append("closed")

    class Agent:
        def invoke(self, *a, **k):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(research, "make_model", lambda: SimpleNamespace(model_name="fake"))
    monkeypatch.setattr(research, "open_sandbox", lambda: Ctx())
    monkeypatch.setattr(research, "upload", lambda backend, files: events.append(sorted(files)))
    monkeypatch.setattr(research, "build_lead_agent", lambda backend, model: Agent())
    assert research.main("topic") == 1
    assert "FAILED" in capsys.readouterr().err and events[-1] == "closed"
    uploaded = [e for e in events if isinstance(e, list)][0]
    assert uploaded == sorted([research.VALIDATOR_PATH, research.FINALIZER_PATH])   # nothing else (no .env) goes in


# ---------- agents ----------
def test_lead_agent_builds_with_todos_and_limits():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    import agents
    specs = agents.build_subagents()
    assert [s["name"] for s in specs] == ["researcher", "citation-checker"]
    assert [t.name for t in specs[0]["tools"]] == [t.name for t in tools.SOURCE_TOOLS]
    assert [t.name for t in specs[1]["tools"]] == ["web_fetch"]
    assert all(s["middleware"] for s in specs)
    for path in (agents.NOTES_DIR, agents.SOURCES_PATH, agents.REPORT_PATH, agents.FINALIZER_PATH, agents.VALIDATOR_PATH):
        assert path in agents.LEAD_PROMPT
    agent = agents.build_lead_agent(None, GenericFakeChatModel(messages=iter([])))
    assert agent is not None
