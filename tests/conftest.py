from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ailab_agents.data import Task

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "fixtures" / "corpus.jsonl"
TASKS = REPO_ROOT / "fixtures" / "tasks.jsonl"
CASSETTE = REPO_ROOT / "fixtures" / "cassettes" / "agent.json"


class OracleProvider:
    """A SYNTHETIC authentic provider for exercising the gate's PASS branch in tests.

    It reports ``name = "ollama"`` (so authenticity passes) and drives a correct trajectory for
    each task: retrieve each gold-fact record, run the calc tool, then answer. It is an in-memory
    test double, never committed data; the real gemma3:27b pass is recorded separately.
    """

    name = "ollama"
    model = "gemma3:27b"

    def __init__(self, tasks: Sequence[Task]) -> None:
        self._by_question = {task.question: task for task in tasks}

    def complete(self, prompt: str) -> str:
        question = ""
        history = 0
        in_history = False
        for line in prompt.splitlines():
            if line.startswith("Question:"):
                question = line[len("Question:") :].strip()
            if line.rstrip() == "History:":
                in_history = True
                continue
            if in_history:
                if line.startswith("Respond"):
                    break
                stripped = line.strip()
                if stripped and stripped != "(none yet)":
                    history += 1
        task = self._by_question[question]
        facts = task.gold_facts
        if "calc" in task.required_tools:
            if history == 0:
                return f"ACTION: retrieve {facts[0]}"
            if history == 1:
                return f"ACTION: retrieve {facts[1]}"
            if history == 2:
                return f"ACTION: calc {task.answer}"
            return f"ANSWER: {task.answer}"
        if history == 0:
            return f"ACTION: retrieve {facts[0]}"
        return f"ANSWER: {task.answer}"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host env vars from leaking into config-sensitive tests."""
    for var in (
        "AILAB_CONFIG",
        "AILAB_LAB",
        "AILAB_CORPUS",
        "AILAB_TASKS",
        "AILAB_PROVIDER",
        "AILAB_CASSETTE",
        "AILAB_MODEL",
        "AILAB_REAL_PROVIDER",
        "AILAB_MAX_STEPS",
        "AILAB_OUTPUT",
        "AILAB_MIN_TASK_SUCCESS",
        "AILAB_MIN_TOOL_PRECISION",
        "AILAB_TOLERANCE",
        "AILAB_INVALID_ACTION_MAX",
        "AILAB_MIN_TASKS",
        "AILAB_PRIMARY_METRIC",
        "AILAB_COMMIT",
        "GITHUB_SHA",
        "GITHUB_STEP_SUMMARY",
    ):
        monkeypatch.delenv(var, raising=False)
