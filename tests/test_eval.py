import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ailab_agents.config import EvalConfig
from ailab_agents.data import DatasetError, load_tasks
from ailab_agents.eval import current_commit, evaluate, main
from ailab_agents.providers import StubProvider
from tests.conftest import CORPUS, TASKS, OracleProvider


def _cfg(**kw: object) -> EvalConfig:
    base: dict[str, object] = {"corpus": CORPUS, "tasks": TASKS}
    base.update(kw)
    return EvalConfig(**base)  # type: ignore[arg-type]


def _oracle() -> OracleProvider:
    return OracleProvider(load_tasks(TASKS))


def test_stub_run_fails_the_gate() -> None:
    result = evaluate(_cfg(min_task_success=0.6, min_tool_precision=0.5), StubProvider())
    assert result["passed"] is False
    assert result["metrics"]["task_success"] == 0.0
    assert any("task_success" in f for f in result["failures"])
    assert any("authenticity" in f for f in result["failures"])
    # The stub still exercises the machinery: it calls a tool and varies its answers.
    assert result["tool_usage"]["total_tool_calls"] > 0
    assert result["non_triviality"]["distinct_answers"] > 1


def test_oracle_run_passes_the_gate() -> None:
    result = evaluate(_cfg(min_task_success=0.6, min_tool_precision=0.5), _oracle())
    assert result["passed"] is True, result["failures"]
    assert result["metrics"]["task_success"] == 1.0
    assert result["metrics"]["tool_precision"] == 1.0
    assert result["provider"] == "ollama"
    assert result["authenticity"]["authentic"] is True


def test_results_record_contract() -> None:
    result = evaluate(_cfg(), _oracle())
    contract: dict[str, type | tuple[type, ...]] = {
        "schema_version": int,
        "lab": str,
        "dataset": str,
        "provider": str,
        "model": str,
        "primary_metric": str,
        "metrics": dict,
        "threshold": float,
        "passed": bool,
        "commit": (str, type(None)),
        "generated_at": str,
    }
    for key, kind in contract.items():
        assert isinstance(result[key], kind), key
    assert result["schema_version"] == 1
    assert result["lab"] == "ailab-agents"
    assert result["primary_metric"] == "task_success"
    assert set(result["metrics"]) == {
        "task_success",
        "tool_precision",
        "invalid_action_rate",
        "mean_steps",
        "n",
    }
    assert len(result["corpus_sha256"]) == 64
    assert datetime.fromisoformat(result["generated_at"]).utcoffset() == timedelta(0)


def test_no_tools_collapses_task_success() -> None:
    # The stub respects the empty tool list and answers blind, so no tools are called.
    result = evaluate(_cfg(min_task_success=0.6), StubProvider(), no_tools=True)
    assert result["passed"] is False
    assert result["tool_usage"]["total_tool_calls"] == 0
    assert any("tool_usage" in f for f in result["failures"])


def test_below_min_tasks_raises(tmp_path: Path) -> None:
    tiny = tmp_path / "t.jsonl"
    tiny.write_text(
        '{"id":"a","question":"q","answer":"1","required_tools":["retrieve"],"gold_facts":["x"]}\n'
    )
    with pytest.raises(DatasetError, match="min_tasks"):
        evaluate(_cfg(tasks=tiny))


def test_empty_tasks_raises(tmp_path: Path) -> None:
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    with pytest.raises(DatasetError, match="empty"):
        evaluate(_cfg(tasks=empty))


def test_main_writes_results_and_reports_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AILAB_COMMIT", "abc1234deadbeef")
    out = tmp_path / "r.json"
    code = main(
        [
            "--corpus", str(CORPUS), "--tasks", str(TASKS),
            "--output", str(out), "--min-task-success", "0.6",
        ]
    )  # fmt: skip
    assert code == 1  # the default stub cannot pass
    record = json.loads(out.read_text())
    assert record["commit"] == "abc1234deadbeef"
    captured = capsys.readouterr()
    assert "task_success" in captured.out
    assert "authenticity:" in captured.out
    assert "REGRESSION:" in captured.err


def test_main_bad_data_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--tasks", str(tmp_path / "nope.jsonl")]) == 2
    assert "error" in capsys.readouterr().err


def test_main_replay_stub_cassette_misses_with_default_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The committed cassette is keyed on the stub model id; replaying with gemma3:27b misses.
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "--provider", "replay",
            "--cassette", "fixtures/cassettes/agent.json",
            "--corpus", str(CORPUS), "--tasks", str(TASKS),
            "--output", str(tmp_path / "r.json"),
        ]
    )  # fmt: skip
    assert code == 2
    assert "error" in capsys.readouterr().err


def test_table_from_existing_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "r.json"
    main(["--corpus", str(CORPUS), "--tasks", str(TASKS), "--output", str(out)])
    capsys.readouterr()
    assert main(["--table-from", str(out)]) == 0
    assert "| stub |" in capsys.readouterr().out
    (tmp_path / "bad.json").write_text("{}")
    assert main(["--table-from", str(tmp_path / "bad.json")]) == 2


def test_writes_github_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    main(
        [
            "--corpus", str(CORPUS), "--tasks", str(TASKS),
            "--output", str(tmp_path / "r.json"),
        ]
    )  # fmt: skip
    assert "### Eval results" in summary.read_text()


def test_commit_null_when_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert current_commit() is None
    monkeypatch.setenv("AILAB_COMMIT", "unknown")
    assert current_commit() is None
