from pathlib import Path
from typing import Any

import pytest

from ailab_agents.config import ConfigError, EvalConfig, load_config


def test_defaults() -> None:
    cfg = EvalConfig()
    assert cfg.lab == "ailab-agents"
    assert cfg.provider == "stub"
    assert cfg.max_steps == 6
    assert cfg.primary_metric_key == "task_success"
    assert cfg.real_provider == "ollama"
    assert cfg.thresholds == {"task_success": 0.0, "tool_precision": 0.0}


def test_config_defaults_when_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_config(env={}) == EvalConfig()


def test_file_then_env_override(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text(
        '[eval]\nprovider = "replay"\nmax_steps = 4\nmodel = "m1"\n'
        "[thresholds]\ntask_success = 0.5\ntool_precision = 0.4\n"
    )
    cfg = load_config(cfg_file, env={"AILAB_MAX_STEPS": "8", "AILAB_MIN_TASK_SUCCESS": "0.7"})
    assert cfg.provider == "replay"
    assert cfg.model == "m1"
    assert cfg.max_steps == 8  # env wins over file
    assert cfg.min_task_success == 0.7
    assert cfg.thresholds["tool_precision"] == 0.4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_task_success": 1.5}, r"\[0, 1\]"),
        ({"tolerance": -0.1}, r"\[0, 1\]"),
        ({"max_steps": 0}, "max_steps"),
        ({"min_tasks": 0}, "min_tasks"),
        ({"provider": "web"}, "provider"),
        ({"primary_metric": "f2"}, "primary_metric"),
    ],
)
def test_range_validation(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        EvalConfig(**kwargs)


def test_env_int_must_parse() -> None:
    with pytest.raises(ConfigError, match="integer"):
        load_config(env={"AILAB_MAX_STEPS": "lots"})


def test_env_float_must_parse() -> None:
    with pytest.raises(ConfigError, match="number"):
        load_config(env={"AILAB_TOLERANCE": "wide"})


def test_missing_explicit_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml", env={})


def test_invalid_toml_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text("[eval\n")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(cfg, env={})


def test_env_points_at_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "c.toml"
    cfg.write_text('[eval]\nprovider = "replay"\n')
    assert load_config(env={"AILAB_CONFIG": str(cfg)}).provider == "replay"


def test_env_path_and_string_overrides(tmp_path: Path) -> None:
    cfg = load_config(
        env={"AILAB_CORPUS": "x/corpus.jsonl", "AILAB_REAL_PROVIDER": "vllm"},
    )
    assert cfg.corpus == Path("x/corpus.jsonl")
    assert cfg.real_provider == "vllm"
