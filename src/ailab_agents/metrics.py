"""Metrics for the agents lab.

The primary metric is ``task_success``: an exact match of the agent's final answer against
the normalized ground truth. ``tool_precision`` scores how many of the agent's tool calls were
among the task's required tools. ``invalid_action_rate`` and ``trajectory_steps`` expose the
shape of the trajectory so a degenerate loop is visible rather than hidden. All functions are
small and unit-tested, and :func:`normalize_answer` is shared with the calc tool so a numeric
answer compares the same way everywhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ailab_agents.agent import Trajectory


def normalize_answer(value: str) -> str:
    """Lowercase, strip, collapse whitespace, and canonicalize a numeric answer.

    A number is normalized so ``"7"``, ``"7.0"`` and ``" 7 "`` all compare equal. A
    non-numeric answer is lowercased and whitespace-collapsed only.
    """
    collapsed = " ".join(value.strip().split()).lower()
    try:
        number = float(collapsed)
    except ValueError:
        return collapsed
    if number.is_integer():
        return str(int(number))
    return repr(number)


def task_success(trajectory: Trajectory, gold: str) -> float:
    """1.0 when the agent's final answer matches the gold answer (normalized), else 0.0."""
    if trajectory.final_answer is None:
        return 0.0
    return 1.0 if normalize_answer(trajectory.final_answer) == normalize_answer(gold) else 0.0


def tool_precision(trajectory: Trajectory, required: Iterable[str]) -> float | None:
    """Fraction of tool calls whose name is in ``required``; ``None`` when no tools were called.

    ``None`` (not 0.0) for a no-call trajectory keeps "the agent used no tools" distinct from
    "the agent used the wrong tools", so the mean is taken only over tasks that called a tool.
    """
    calls = trajectory.tool_calls
    if not calls:
        return None
    allowed = set(required)
    return sum(1 for name in calls if name in allowed) / len(calls)


def trajectory_steps(trajectory: Trajectory) -> int:
    """Number of model steps the agent took for this task."""
    return len(trajectory.steps)


def invalid_action_rate(trajectory: Trajectory) -> float:
    """Fraction of steps whose parsed action was invalid (0.0 for an empty trajectory)."""
    steps = trajectory.steps
    if not steps:
        return 0.0
    invalid = sum(1 for step in steps if step.action_kind == "invalid")
    return invalid / len(steps)


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence."""
    return sum(values) / len(values) if values else None
