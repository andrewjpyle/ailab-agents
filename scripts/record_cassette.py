#!/usr/bin/env python3
"""Record a REAL agent cassette from a live Ollama model.

It drives the exact bounded loop the eval runs: for every task it asks the model each step
prompt (built by :func:`ailab_agents.agent.build_prompt`), dispatches the tool the model
chooses, and records EVERY step prompt keyed by ``sha256(model, prompt)`` so
``ReplayProvider`` can reproduce the trajectory later with no network.

The model id comes from ``AILAB_RECORD_MODEL`` (default ``gemma3:27b``). The server URL comes
from ``OLLAMA_HOST`` and is NEVER written to disk: the cassette stores only the prompt hash and
the response. A scan refuses to write if any host, IP or ``/Users/`` path leaked into a response.

Usage::

    OLLAMA_HOST=http://<host>:11434 make record
    OLLAMA_HOST=http://<host>:11434 AILAB_RECORD_MODEL=gemma3:27b \
        uv run python scripts/record_cassette.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ailab_agents.agent import AgentConfig, build_prompt, parse_action  # noqa: E402
from ailab_agents.config import load_config  # noqa: E402
from ailab_agents.data import load_corpus, load_tasks  # noqa: E402
from ailab_agents.providers import OllamaProvider, cassette_key  # noqa: E402
from ailab_agents.tools import ToolError, build_registry  # noqa: E402

# Default to a model-tagged path so the committed stub cassette (agent.json) is never clobbered.
CASSETTE = REPO / "fixtures" / "cassettes" / "agent_gemma3_27b.json"

# A recorded response must never carry a host, a tailnet address, or a local filesystem path.
_LEAK = re.compile(
    r"(\d{1,3}(?:\.\d{1,3}){3})|(\bhttps?://)|(/Users/)|(\.ts\.net\b)",
    re.IGNORECASE,
)


def _get_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out: dict[str, Any] = json.loads(resp.read())
    return out


def main() -> int:
    if not os.environ.get("OLLAMA_HOST"):
        print("record_cassette: set OLLAMA_HOST (never hardcoded here)", file=sys.stderr)
        return 2
    model = os.environ.get("AILAB_RECORD_MODEL", "gemma3:27b")
    cfg = load_config()
    records = load_corpus(cfg.corpus)
    tasks = load_tasks(cfg.tasks)
    registry = build_registry(records)
    provider = OllamaProvider(model)
    agent_config = AgentConfig(max_steps=cfg.max_steps)

    # Provenance metadata (no host): Ollama version and the model digest, so a re-record is
    # identifiable and reproducible. Failures here are non-fatal.
    version = ""
    digest = ""
    try:
        version = str(_get_json(f"{provider.base}/api/version").get("version", ""))
        tags = _get_json(f"{provider.base}/api/tags").get("models", [])
        digest = next((str(m.get("digest", "")) for m in tags if m.get("name") == model), "")
    except Exception as exc:
        print(f"record_cassette: metadata lookup failed ({exc}); continuing", file=sys.stderr)

    entries: dict[str, dict[str, Any]] = {}
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {task.id}: driving the loop with {model} ...", flush=True)
        history: list[tuple[str, str]] = []
        for _ in range(agent_config.max_steps):
            prompt = build_prompt(task.question, registry.list(), history)
            raw = provider.generate(prompt)
            response = str(raw.get("response", ""))
            if not response.strip():
                print(f"record_cassette: empty response for {task.id}; aborting", file=sys.stderr)
                return 1
            if _LEAK.search(response):
                print(
                    f"record_cassette: a host/IP/path leaked into the {task.id} response; "
                    "refusing to write the cassette",
                    file=sys.stderr,
                )
                return 1
            entries[cassette_key(model, prompt)] = {
                "model": model,
                "response": response,
                "prompt_eval_count": int(raw.get("prompt_eval_count", 0) or 0),
                "eval_count": int(raw.get("eval_count", 0) or 0),
            }
            kind, name, arg = parse_action(response)
            if kind == "answer":
                break
            if kind == "tool":
                assert name is not None
                try:
                    obs = registry.call(name, arg or "")
                except ToolError as exc:
                    obs = f"ERROR: {exc}"
                history.append((f"{name} {arg or ''}".strip(), obs))

    payload = {
        "provider": "ollama",
        "model": model,
        "model_digest": digest,
        "ollama_version": version,
        "prompt_format_version": 1,
        "temperature": 0,
        "seed": OllamaProvider.DEFAULT_SEED,
        "recorded_on": datetime.now(UTC).date().isoformat(),
        "note": (
            "Real recording from a local Ollama server (host from OLLAMA_HOST, never stored). "
            "Keys are sha256(model, prompt) built from the canonical bounded agent loop. "
            "Pinned temperature 0 and a fixed seed make a re-record deterministic."
        ),
        "entries": entries,
    }
    blob = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if _LEAK.search(blob):
        print("record_cassette: leak scan failed on the assembled cassette", file=sys.stderr)
        return 1
    CASSETTE.parent.mkdir(parents=True, exist_ok=True)
    CASSETTE.write_text(blob, encoding="utf-8")
    print(f"record_cassette: wrote {len(entries)} entries for model {model} to {CASSETTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
