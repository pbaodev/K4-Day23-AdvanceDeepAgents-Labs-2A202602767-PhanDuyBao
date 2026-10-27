"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_CITATION = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]; not [3](link)
_REF_LINE = re.compile(r"^\s*\[(\d+)\]")
_URL = re.compile(r"https?://[^\s<>]+")


def _expand(group):
    """'1, 2' -> [1, 2];  '1-3' -> [1, 2, 3]."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[\u2013-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def _cited_numbers(body):
    """Every n cited as [n] / [n, m] / [n-m] in prose; code spans and Markdown links [n](url) do not count."""
    prose = "".join(seg for i, seg in enumerate(_CODE.split(body)) if i % 2 == 0)
    return {n for match in _CITATION.finditer(prose) for n in _expand(match.group(1))}


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]
    problems = []
    by_n = {}
    url_count = Counter()
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry is not an object: {entry!r}")
            continue
        n, url = entry.get("n"), entry.get("url")
        if isinstance(n, int) and not isinstance(n, bool):
            if n in by_n:
                problems.append(f"source number [{n}] appears twice in sources.json")
            by_n.setdefault(n, entry)
        else:
            problems.append(f"source n={n!r} is not an integer")
        if isinstance(url, str) and re.match(r"https?://", url):
            url_count[url] += 1
        else:
            problems.append(f"source [{n}] has an invalid url: {url!r}")
    problems += [f"url used by more than one source: {url}" for url, count in url_count.items() if count > 1]

    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        problems.append("missing the '## References' heading")
        body, references = report_text, ""
    else:
        body, references = report_text[:headings[-1].start()], report_text[headings[-1].end():]

    cited = _cited_numbers(body)
    problems += [f"[{n}] cited but missing from sources.json" for n in sorted(cited - set(by_n))]
    problems += [f"source [{n}] never cited" for n in sorted(set(by_n) - cited)]

    if headings:
        lines_by_n = {}
        for line in references.splitlines():
            match = _REF_LINE.match(line)
            if match:
                lines_by_n.setdefault(int(match.group(1)), []).append(line)
        for n in sorted(by_n):
            if n not in lines_by_n:
                problems.append(f"source [{n}] has no line in ## References")
        for n, lines in sorted(lines_by_n.items()):
            if n not in by_n:
                problems.append(f"## References has a line [{n}] that is not in sources.json")
            if len(lines) > 1:
                problems.append(f"## References has {len(lines)} lines for [{n}] (need exactly one)")
            for line in lines[:1]:
                urls = [u.rstrip(".,;") for u in _URL.findall(line)]
                if len(urls) != 1:
                    problems.append(f"reference [{n}] must hold exactly one URL, found {len(urls)}")
                elif n in by_n and urls[0] != by_n[n].get("url"):
                    problems.append(f"reference [{n}] URL {urls[0]} differs from sources.json ({by_n[n].get('url')})")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
