"""A bounded ReAct-style agent: list tools, call them, observe, answer.

The agent is deliberately small and robust. Each step it builds a prompt that lists the
available tools and the observation history, asks the provider for the next line, and parses
ONE action from it:

* ``ACTION: <tool_name> <arg>`` calls a tool and appends its observation to the history.
* ``ANSWER: <final answer>`` ends the loop.
* anything else is an INVALID action; it is counted and the loop continues (bounded by
  ``max_steps``), so a model that never emits a clean line cannot hang the eval.

The loop is bounded so an agent that never answers stops at ``max_steps`` rather than running
forever. The :class:`Trajectory` records every step, so the eval can score tool use, invalid
actions and the final answer without re-running the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ailab_agents.tools import ToolError, ToolRegistry

ACTION_PREFIX = "ACTION:"
ANSWER_PREFIX = "ANSWER:"


@dataclass(frozen=True, slots=True)
class Step:
    """One model step: the prompt it saw, the raw line it returned, and what came of it."""

    prompt: str
    raw_output: str
    action_kind: str  # "tool" | "answer" | "invalid"
    tool_name: str | None
    tool_arg: str | None
    observation: str | None


@dataclass(frozen=True, slots=True)
class Trajectory:
    """The full record of one task run."""

    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    stopped_reason: str = "max_steps"  # "answered" | "max_steps"
    tool_calls: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Loop bounds and modes. ``no_tools`` hides every tool, so the agent must answer blind."""

    max_steps: int = 6
    no_tools: bool = False


@runtime_checkable
class Provider(Protocol):
    """Structural type for a text-completion backend (see ailab_core.providers.LLMProvider)."""

    name: str
    model: str

    def complete(self, prompt: str) -> str: ...


def parse_action(raw: str) -> tuple[str, str | None, str | None]:
    """Parse one action from a model output. Returns ``(kind, tool_name, arg)``.

    The first line that starts with ``ANSWER:`` or ``ACTION:`` wins; ANSWER is preferred when
    both appear. A missing tool name after ``ACTION:`` is an invalid action, not a crash.
    """
    answer: str | None = None
    action: tuple[str, str] | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if answer is None and stripped.startswith(ANSWER_PREFIX):
            answer = stripped[len(ANSWER_PREFIX) :].strip()
        elif action is None and stripped.startswith(ACTION_PREFIX):
            body = stripped[len(ACTION_PREFIX) :].strip()
            parts = body.split(None, 1)
            if parts:
                action = (parts[0], parts[1].strip() if len(parts) > 1 else "")
    if answer is not None:
        return ("answer", None, answer)
    if action is not None:
        return ("tool", action[0], action[1])
    return ("invalid", None, None)


def build_prompt(question: str, tools: list[dict[str, str]], history: list[tuple[str, str]]) -> str:
    """Build the step prompt: the question, the available tools, and the observation history."""
    if tools:
        tool_lines = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
    else:
        tool_lines = "- (no tools available)"
    if history:
        history_lines = "\n".join(f"{i}. {act} -> {obs}" for i, (act, obs) in enumerate(history, 1))
    else:
        history_lines = "(none yet)"
    return (
        "You are a tool-using agent. Answer the question by calling tools one at a time, then "
        "give a final answer.\n\n"
        f"Question: {question}\n\n"
        f"Tools:\n{tool_lines}\n\n"
        f"History:\n{history_lines}\n\n"
        "Respond with exactly one line, either:\n"
        "ACTION: <tool_name> <arg>\n"
        "ANSWER: <final answer>\n"
    )


def run_agent(
    task_id: str,
    question: str,
    provider: Provider,
    registry: ToolRegistry,
    config: AgentConfig,
) -> Trajectory:
    """Run the bounded loop for one task and return its :class:`Trajectory`."""
    tools = [] if config.no_tools else registry.list()
    history: list[tuple[str, str]] = []
    steps: list[Step] = []
    tool_calls: list[str] = []
    final_answer: str | None = None
    stopped_reason = "max_steps"

    for _ in range(config.max_steps):
        prompt = build_prompt(question, tools, history)
        raw = provider.complete(prompt)
        kind, name, arg = parse_action(raw)

        if kind == "answer":
            final_answer = arg
            stopped_reason = "answered"
            steps.append(Step(prompt, raw, "answer", None, arg, None))
            break

        if kind == "tool":
            assert name is not None
            tool_calls.append(name)
            try:
                observation = registry.call(name, arg or "")
            except ToolError as exc:
                observation = f"ERROR: {exc}"
            history.append((f"{name} {arg or ''}".strip(), observation))
            steps.append(Step(prompt, raw, "tool", name, arg, observation))
            continue

        steps.append(Step(prompt, raw, "invalid", None, None, None))

    return Trajectory(
        task_id=task_id,
        steps=steps,
        final_answer=final_answer,
        stopped_reason=stopped_reason,
        tool_calls=tool_calls,
    )
