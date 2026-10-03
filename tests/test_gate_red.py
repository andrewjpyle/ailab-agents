"""Proof-of-gate: the gate must FAIL on each sabotage, naming the failing metric.

This mirrors ``make red``. Each case asserts the exit code and that the failing metric is named
in stderr, so a future change that quietly weakens the gate is caught.
"""

from pathlib import Path

import pytest

from ailab_agents.eval import main
from tests.conftest import CORPUS, TASKS

BASE = ["--corpus", str(CORPUS), "--tasks", str(TASKS)]


def _run(args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.chdir(tmp_path)
    return main([*BASE, "--output", str(tmp_path / "r.json"), "--min-task-success", "0.6", *args])


def test_red_1_stub_below_task_success_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run([], tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1
    assert "REGRESSION: task_success" in err


def test_red_2_no_tools_trips_tool_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(["--no-tools"], tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1
    assert "tool_usage" in err


def test_red_3_hardcode_answer_trips_non_triviality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(["--hardcode-answer"], tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1
    assert "non_triviality" in err


def test_red_4_force_invalid_trips_invalid_action_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(["--force-invalid"], tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 1
    assert "invalid_action_rate" in err


def test_red_5_empty_tasks_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert main(["--corpus", str(CORPUS), "--tasks", str(empty)]) == 2
    assert "empty" in capsys.readouterr().err


def test_red_6_below_min_tasks_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    tiny = tmp_path / "tiny.jsonl"
    tiny.write_text(
        '{"id":"a","question":"q","answer":"1","required_tools":["retrieve"],"gold_facts":["x"]}\n'
    )
    assert main(["--corpus", str(CORPUS), "--tasks", str(tiny)]) == 2
    assert "min_tasks" in capsys.readouterr().err
