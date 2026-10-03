from __future__ import annotations

from ailab_agents.agent import (
    AgentConfig,
    build_prompt,
    parse_action,
    run_agent,
)
from ailab_agents.data import Record
from ailab_agents.providers import StubProvider
from ailab_agents.tools import ToolRegistry, build_registry

RECORDS = [
    Record(id="alpha-base", text="Alpha Base keeps a crew of 10 and 3 sensor arrays."),
    Record(id="bravo-base", text="Bravo Base keeps a crew of 20 and 5 sensor arrays."),
]


def test_parse_answer() -> None:
    assert parse_action("ANSWER: 42") == ("answer", None, "42")


def test_parse_tool_with_arg() -> None:
    assert parse_action("ACTION: calc 1+2") == ("tool", "calc", "1+2")


def test_parse_tool_without_arg() -> None:
    assert parse_action("ACTION: retrieve") == ("tool", "retrieve", "")


def test_parse_invalid_when_no_keyword() -> None:
    assert parse_action("I think it is probably 5") == ("invalid", None, None)


def test_parse_bare_action_is_invalid() -> None:
    assert parse_action("ACTION:") == ("invalid", None, None)


def test_parse_prefers_answer_over_action() -> None:
    assert parse_action("ACTION: calc 1+1\nANSWER: 2") == ("answer", None, "2")


def test_build_prompt_lists_tools_and_history() -> None:
    tools = [{"name": "calc", "description": "do math"}]
    prompt = build_prompt("How much?", tools, [("calc 1+1", "2")])
    assert "Question: How much?" in prompt
    assert "- calc: do math" in prompt
    assert "1. calc 1+1 -> 2" in prompt


def test_build_prompt_marks_absent_tools_and_history() -> None:
    prompt = build_prompt("Q?", [], [])
    assert "(no tools available)" in prompt
    assert "(none yet)" in prompt


def test_run_agent_stub_retrieves_then_answers() -> None:
    registry = build_registry(RECORDS)
    traj = run_agent("t", "alpha base crew", StubProvider(), registry, AgentConfig())
    assert traj.tool_calls == ["retrieve"]
    assert traj.stopped_reason == "answered"
    assert traj.final_answer is not None
    assert [s.action_kind for s in traj.steps] == ["tool", "answer"]


def test_run_agent_no_tools_answers_blind() -> None:
    registry = build_registry(RECORDS)
    traj = run_agent("t", "the question", StubProvider(), registry, AgentConfig(no_tools=True))
    assert traj.tool_calls == []
    assert traj.final_answer == "the question"


def test_run_agent_records_tool_error_observation() -> None:
    registry = build_registry(RECORDS)

    class _BadRetrieve:
        name = "ollama"
        model = "gemma3:27b"

        def __init__(self) -> None:
            self._done = False

        def complete(self, prompt: str) -> str:
            if not self._done:
                self._done = True
                return "ACTION: retrieve zzzznothing"
            return "ANSWER: done"

    traj = run_agent("t", "q", _BadRetrieve(), registry, AgentConfig())
    assert traj.tool_calls == ["retrieve"]
    assert traj.steps[0].observation is not None
    assert traj.steps[0].observation.startswith("ERROR:")


def test_run_agent_invalid_then_bounded_stop() -> None:
    class _AlwaysInvalid:
        name = "stub"
        model = "stub-agent-v1"

        def complete(self, prompt: str) -> str:
            return "no clean line here"

    traj = run_agent("t", "q", _AlwaysInvalid(), ToolRegistry(), AgentConfig(max_steps=3))
    assert traj.stopped_reason == "max_steps"
    assert traj.final_answer is None
    assert len(traj.steps) == 3
    assert all(s.action_kind == "invalid" for s in traj.steps)
