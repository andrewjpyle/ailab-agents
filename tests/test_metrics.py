from __future__ import annotations

from ailab_agents.agent import Step, Trajectory
from ailab_agents.metrics import (
    invalid_action_rate,
    mean,
    normalize_answer,
    task_success,
    tool_precision,
    trajectory_steps,
)


def _traj(final: str | None, tool_calls: list[str], kinds: list[str]) -> Trajectory:
    steps = [Step("p", "o", kind, None, None, None) for kind in kinds]
    return Trajectory(task_id="t", steps=steps, final_answer=final, tool_calls=tool_calls)


def test_normalize_numbers_and_text() -> None:
    assert normalize_answer("7") == normalize_answer("7.0") == "7"
    assert normalize_answer("  7 ") == "7"
    assert normalize_answer("7.50") == "7.5"
    assert normalize_answer("Yes  Please") == "yes please"
    assert normalize_answer("-0") == "0"


def test_task_success_match_miss_and_none() -> None:
    assert task_success(_traj("79", ["calc"], ["answer"]), "79") == 1.0
    assert task_success(_traj("79.0", ["calc"], ["answer"]), "79") == 1.0
    assert task_success(_traj("80", ["calc"], ["answer"]), "79") == 0.0
    assert task_success(_traj(None, [], ["invalid"]), "79") == 0.0


def test_tool_precision_fraction_and_none() -> None:
    assert tool_precision(_traj("x", ["retrieve", "calc"], ["tool"]), ["retrieve", "calc"]) == 1.0
    assert tool_precision(_traj("x", ["retrieve", "web"], ["tool"]), ["retrieve", "calc"]) == 0.5
    assert tool_precision(_traj("x", [], ["answer"]), ["retrieve"]) is None


def test_trajectory_steps_and_invalid_rate() -> None:
    traj = _traj("x", ["calc"], ["tool", "invalid", "answer"])
    assert trajectory_steps(traj) == 3
    assert invalid_action_rate(traj) == 1 / 3
    assert invalid_action_rate(_traj("x", [], [])) == 0.0


def test_mean_handles_empty() -> None:
    assert mean([]) is None
    assert mean([1.0, 0.0]) == 0.5
