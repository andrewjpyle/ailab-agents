import json
from pathlib import Path
from typing import Any

from ailab_agents.compare import changed_variables, main, metric_deltas, render


def _record(max_steps: int, task_success: float, provider: str = "ollama") -> dict[str, Any]:
    return {
        "provider": provider,
        "passed": task_success >= 0.6,
        "config": {"max_steps": max_steps, "provider": provider, "min_tasks": 8},
        "metrics": {
            "task_success": task_success,
            "tool_precision": 1.0,
            "invalid_action_rate": 0.0,
            "mean_steps": 3.0,
            "n": 20,
        },
    }


def test_changed_variables_single() -> None:
    changed = changed_variables(_record(4, 0.5), _record(6, 0.9))
    assert changed == {"max_steps": (4, 6)}


def test_metric_deltas_skip_n() -> None:
    rows = metric_deltas(_record(4, 0.50), _record(6, 0.90))
    names = {r[0] for r in rows}
    assert "n" not in names
    success = next(r for r in rows if r[0] == "task_success")
    assert round(success[3], 2) == 0.40


def test_render_names_single_variable() -> None:
    out = render(_record(4, 0.5), _record(6, 0.9))
    assert "single changed variable: max_steps: 4 -> 6" in out
    assert "task_success" in out


def test_render_rejects_multi_variable() -> None:
    a = _record(4, 0.5)
    b = _record(6, 0.9)
    b["config"]["min_tasks"] = 12
    assert "ERROR" in render(a, b)


def test_main_roundtrip(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(_record(4, 0.5)))
    b.write_text(json.dumps(_record(6, 0.9)))
    assert main([str(a), str(b)]) == 0
    assert "max_steps" in capsys.readouterr().out


def test_main_usage_error() -> None:
    assert main(["only-one.json"]) == 2
