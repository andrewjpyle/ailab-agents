"""LLM provider seam for the agent brain.

The discipline-agnostic pieces (the ``LLMProvider`` protocol, ``cassette_key``,
``CassetteMissError``, ``ReplayProvider`` and ``OllamaProvider``) live in ``ailab_core`` and are
re-exported here so existing imports keep working. Only :class:`StubProvider` is agent-specific:
its output is a next-action line, so it cannot be shared across labs.

:class:`StubProvider` is a deterministic, offline agent brain that is FORMAT-valid but
TASK-naive. It emits well-formed ``ACTION:`` / ``ANSWER:`` lines, so the loop, the parser, the
tool dispatch and the metrics are all exercised with no network. But it does NOT chain
retrieve -> calc: it does one retrieve and then answers with the raw retrieved text, never doing
the arithmetic. So ``task_success`` is LOW with the stub, which is what keeps the gate honest:
the stub scores below the floor and a green gate needs the real recorded model.
"""

from __future__ import annotations

from ailab_core.providers import (
    CassetteMissError,
    LLMProvider,
    OllamaProvider,
    ReplayProvider,
    SupportsTokenCounts,
    cassette_key,
)

_QUESTION_PREFIX = "Question:"
_HISTORY_HEADER = "History:"
_RESPOND_PREFIX = "Respond"
_NO_TOOLS_MARKER = "(no tools available)"
_NO_HISTORY_MARKER = "(none yet)"


def _extract_question(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(_QUESTION_PREFIX):
            return line[len(_QUESTION_PREFIX) :].strip()
    return ""


def _history_entries(prompt: str) -> list[str]:
    """Return the numbered history lines (empty when the history is still empty)."""
    entries: list[str] = []
    collecting = False
    for line in prompt.splitlines():
        if line.rstrip() == _HISTORY_HEADER:
            collecting = True
            continue
        if collecting:
            if line.startswith(_RESPOND_PREFIX):
                break
            stripped = line.strip()
            if stripped and stripped != _NO_HISTORY_MARKER:
                entries.append(stripped)
    return entries


class StubProvider:
    """Deterministic offline agent brain. No network, no key. Format-valid but task-naive."""

    name = "stub"
    model = "stub-agent-v1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        question = _extract_question(prompt)
        tools_available = _NO_TOOLS_MARKER not in prompt
        history = _history_entries(prompt)

        if not tools_available:
            # No tools: answer blind from the question (wrong, but task-distinct).
            return f"ANSWER: {question}"
        if not history:
            # First move: always retrieve on the raw question text.
            return f"ACTION: retrieve {question}"
        # Already retrieved once: answer with the raw retrieved text, never using calc.
        last = history[-1]
        observation = last.split(" -> ", 1)[-1].strip() if " -> " in last else last
        return f"ANSWER: {observation}"


__all__ = [
    "CassetteMissError",
    "LLMProvider",
    "OllamaProvider",
    "ReplayProvider",
    "StubProvider",
    "SupportsTokenCounts",
    "cassette_key",
]
