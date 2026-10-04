"""Capture real command output into provenanced JSON for the README graphics.

Each capture records the exact argv, the real combined stdout+stderr, the exit code, the HEAD
commit and a UTC timestamp. The data graphics (anatomy, red-gate) are built only from these files
via readme_kit.load_capture, so a number can never be typed in by hand.

    uv run python docs/assets/src/capture.py

Runs the three commands below from the repo root. Output is captured unbuffered so stdout and
stderr interleave in their true order. Volatile per-run temp paths from the proof-of-gate script
(macOS mktemp) are redacted to <tmpdir> so the committed capture is deterministic and carries no
local filesystem path; nothing else is altered.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAPS = HERE / "captures"

# A macOS mktemp dir looks like /var/folders/xx/.../T/tmp.XXXXXXXX/. Strip the absolute prefix so
# the committed capture shows only <tmpdir>/<file>, never a machine-local path.
TMPDIR_RE = re.compile(r"\S*/tmp\.[A-Za-z0-9]+/")


def head_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def run(name: str, argv: list[str], note: str = "", redact_tmp: bool = False) -> dict:
    # Unbuffered + a single merged stream so stdout and stderr interleave in their true order.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    merged = subprocess.run(
        argv, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    output = merged.stdout
    if redact_tmp:
        output = TMPDIR_RE.sub("<tmpdir>/", output)
    return {
        "name": name,
        "command": argv,
        "exit_code": merged.returncode,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "commit": head_commit(),
        "line_count": output.count("\n"),
        "env_passed": [],
        "note": note,
        "output": output,
    }


def main() -> None:
    CAPS.mkdir(parents=True, exist_ok=True)
    captures = [
        run("demo", ["uv", "run", "ailab-demo"]),
        run(
            "eval",
            [
                "uv", "run", "ailab-eval",
                "--provider", "replay",
                "--cassette", "fixtures/cassettes/agent_gemma3_27b.json",
                "--model", "gemma3:27b",
            ],
            note="real gemma3:27b pass replayed from the committed cassette; no live model called.",
        ),
        run(
            "red",
            ["bash", "scripts/prove_gate.sh"],
            note="Volatile per-run temp paths in the exit-2 error lines redacted to <tmpdir> (macOS mktemp); no other change.",
            redact_tmp=True,
        ),
    ]
    for cap in captures:
        path = CAPS / f"{cap['name']}.json"
        path.write_text(json.dumps(cap, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path} (exit {cap['exit_code']}, {cap['line_count']} lines)")


if __name__ == "__main__":
    main()
