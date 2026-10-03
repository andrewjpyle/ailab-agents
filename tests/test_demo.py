from pathlib import Path

import pytest

from ailab_agents.demo import main as demo_main
from tests.conftest import CORPUS, TASKS


def test_demo_prints_stages_and_runs_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AILAB_CORPUS", str(CORPUS))
    monkeypatch.setenv("AILAB_TASKS", str(TASKS))
    code = demo_main(
        [
            "--corpus", str(CORPUS), "--tasks", str(TASKS),
            "--output", str(tmp_path / "r.json"), "--min-task-success", "0.6",
        ]
    )  # fmt: skip
    # The stub cannot pass, so the embedded eval returns a regression exit code.
    assert code == 1
    out = capsys.readouterr().out
    assert "ailab-agents demo" in out
    assert "ACTION retrieve" in out
    assert "final:" in out
    assert "== eval ==" in out
