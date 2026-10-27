"""research.py - The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    return re.sub(r"\W+", "-", str(topic).lower()).strip("-")[:60].strip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\n\n"
            "Produce a cited survey report on this topic by following your procedure: plan with write_todos, delegate the "
            "sub-questions to researcher subagents in parallel, merge their notes into sources.json, write the report body, "
            "run the finalizer and the validator until it prints OK, and have the citation-checker spot-check a few claims.")


def _text(message, key, default=None):
    return message.get(key, default) if isinstance(message, dict) else getattr(message, key, default)


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Lead messages only: subagent tokens are not included, so this undercounts the real cost.
    """
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in _text(message, "tool_calls") or []:
            calls[_text(call, "name")] += 1
        usage = _text(message, "usage_metadata") or {}
        tokens["input"] += int(usage.get("input_tokens") or 0)
        tokens["output"] += int(usage.get("output_tokens") or 0)
    return {"model": model_name, "elapsed_s": round(elapsed, 1), "subagent_calls": calls.get("task", 0),
            "tool_calls": dict(calls), "tokens": tokens}


# URL shape each source family must have (the grader cross-checks it against sources.json)
FAMILY_URL = {"arxiv": r"https://arxiv\.org/abs/\S+", "hf-daily": r"https://huggingface\.co/papers/\S+",
              "hf-search": r"https://huggingface\.co/papers/\S+"}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    Everything is validated first and the files are written last, so a failed run never leaves an empty or
    half-written report behind (RuntimeError, nothing written).
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, sources_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"the agent did not produce a report ({REPORT_PATH} is missing or empty)")
    if not sources_bytes:
        raise RuntimeError(f"the agent did not produce {SOURCES_PATH}")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except ValueError as exc:   # UnicodeDecodeError and JSONDecodeError are both ValueErrors
        raise RuntimeError(f"the downloaded files are not valid: {exc}") from exc
    if not isinstance(sources, list) or not all(isinstance(entry, dict) for entry in sources):
        raise RuntimeError("sources.json must be a JSON list of objects")

    for entry in sources:
        pattern = FAMILY_URL.get(entry.get("source"))
        if pattern and not re.fullmatch(pattern, str(entry.get("url", ""))):
            print(f"warning: source [{entry.get('n')}] is labelled {entry['source']} but its url is {entry.get('url')}",
                  file=sys.stderr)

    slug = slugify(topic)
    meta = {"topic": topic.strip(), **summarize(messages, elapsed, model_name), "n_sources": len(sources),
            "source_families": sorted({str(entry["source"]) for entry in sources if entry.get("source")})}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{slug}.sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = reports_dir / f"{slug}.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    if not topic.strip():
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__
        start = time.monotonic()
        with open_sandbox() as backend:   # the sandbox is always stopped and removed, even on errors
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": 1000})
            report_path = save_outputs(backend, topic, result.get("messages", []), time.monotonic() - start, str(model_name))
    except Exception as exc:  # noqa: BLE001 - a failed run reports and exits 1, it never leaves a report behind
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
