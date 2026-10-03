#!/usr/bin/env python3
"""Assert every TAGGED lab figure in docs/LEARNING.md equals the committed results.

A lab figure is a number followed by an HTML-comment tag naming its source, for example::

    recall@5 is 0.9643 <!--lab:eval_results.json:metrics.recall@5-->

The checker reads the number, resolves ``metrics.recall@5`` in ``eval_results.json``, and
fails if they differ (compared at the number of decimals written). Untagged numbers are
treated as prose and ignored, so general claims are not policed, only figures that claim
to come from a run. This keeps the learning doc honest: the numbers are generated, never
hand-typed, and drift fails CI.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
LEARNING = REPO / "docs" / "LEARNING.md"

TAG = re.compile(r"(-?\d+(?:\.\d+)?)\s*<!--\s*lab:([^:]+):([^>]+?)\s*-->")


def resolve(data: Any, path: str) -> Any:
    node = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"path {path!r} not found at segment {key!r}")
        node = node[key]
    return node


def main() -> int:
    if not LEARNING.is_file():
        print(f"check_learning_numbers: {LEARNING} not found", file=sys.stderr)
        return 2
    text = LEARNING.read_text(encoding="utf-8")
    cache: dict[str, Any] = {}
    checked = 0
    failures: list[str] = []
    for written, filename, path in TAG.findall(text):
        if filename not in cache:
            cache[filename] = json.loads((REPO / filename).read_text(encoding="utf-8"))
        try:
            actual = resolve(cache[filename], path)
        except KeyError as exc:
            failures.append(f"{filename}:{path}: {exc}")
            continue
        if actual is None:
            failures.append(f"{filename}:{path}: value is null but doc wrote {written}")
            continue
        decimals = len(written.split(".")[1]) if "." in written else 0
        if round(float(actual), decimals) != float(written):
            failures.append(
                f"{filename}:{path}: doc says {written}, results say "
                f"{round(float(actual), decimals)}"
            )
        checked += 1
    if failures:
        print("check_learning_numbers: FAIL", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    if checked == 0:
        print("check_learning_numbers: no tagged lab figures found", file=sys.stderr)
        return 1
    print(f"check_learning_numbers: OK: {checked} tagged figure(s) match the committed results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
