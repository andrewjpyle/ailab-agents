"""Agent eval configuration: a TOML file, overridable by environment variables (12-factor).

Layering is defaults <- TOML file <- environment. The CLI adds a final override layer in
:mod:`ailab_agents.eval`. Ranges are validated up front so a bad config fails loudly.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = "eval_config.toml"

PROVIDERS = ("stub", "replay")
METRIC_BASES = ("task_success", "tool_precision")

#: environment variable -> EvalConfig attribute
ENV_OVERRIDES: dict[str, str] = {
    "AILAB_LAB": "lab",
    "AILAB_CORPUS": "corpus",
    "AILAB_TASKS": "tasks",
    "AILAB_PROVIDER": "provider",
    "AILAB_CASSETTE": "cassette",
    "AILAB_MODEL": "model",
    "AILAB_REAL_PROVIDER": "real_provider",
    "AILAB_MAX_STEPS": "max_steps",
    "AILAB_OUTPUT": "output",
    "AILAB_MIN_TASK_SUCCESS": "min_task_success",
    "AILAB_MIN_TOOL_PRECISION": "min_tool_precision",
    "AILAB_TOLERANCE": "tolerance",
    "AILAB_INVALID_ACTION_MAX": "invalid_action_max",
    "AILAB_MIN_TASKS": "min_tasks",
    "AILAB_PRIMARY_METRIC": "primary_metric",
}

_INT_ATTRS = {"max_steps", "min_tasks"}
_FLOAT_ATTRS = {
    "min_task_success",
    "min_tool_precision",
    "tolerance",
    "invalid_action_max",
}
_PATH_ATTRS = {"corpus", "tasks", "cassette", "output"}


class ConfigError(ValueError):
    """Raised for a missing or invalid configuration value."""


@dataclass(frozen=True, slots=True)
class EvalConfig:
    lab: str = "ailab-agents"
    corpus: Path = Path("fixtures/corpus.jsonl")
    tasks: Path = Path("fixtures/tasks.jsonl")
    provider: str = "stub"
    cassette: Path = Path("fixtures/cassettes/agent.json")
    model: str = "gemma3:27b"
    real_provider: str = "ollama"
    max_steps: int = 6
    output: Path = Path("eval_results.json")
    min_task_success: float = 0.0
    min_tool_precision: float = 0.0
    tolerance: float = 0.05
    invalid_action_max: float = 0.20
    min_tasks: int = 8
    primary_metric: str = "task_success"
    seed: int = 0
    thresholds: dict[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in _FLOAT_ATTRS:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} must be within [0, 1], got {value}")
        if self.max_steps < 1:
            raise ConfigError(f"max_steps must be at least 1, got {self.max_steps}")
        if self.min_tasks < 1:
            raise ConfigError(f"min_tasks must be at least 1, got {self.min_tasks}")
        if self.provider not in PROVIDERS:
            raise ConfigError(f"provider must be one of {PROVIDERS}, not {self.provider!r}")
        if self.primary_metric not in METRIC_BASES:
            raise ConfigError(
                f"primary_metric must be one of {METRIC_BASES}, not {self.primary_metric!r}"
            )
        object.__setattr__(
            self,
            "thresholds",
            {
                "task_success": self.min_task_success,
                "tool_precision": self.min_tool_precision,
            },
        )

    @property
    def primary_metric_key(self) -> str:
        """The primary metric as it appears in the metrics dict (e.g. ``task_success``)."""
        return self.primary_metric


def _int(name: str, raw: Any) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _from_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    section = data.get("eval", {})
    thresholds = data.get("thresholds", {})
    values: dict[str, Any] = {}
    for key in ("lab", "provider", "model", "real_provider", "primary_metric"):
        if key in section:
            values[key] = str(section[key])
    for key in ("corpus", "tasks", "cassette", "output"):
        if key in section:
            values[key] = Path(section[key])
    for key in ("max_steps", "min_tasks"):
        if key in section:
            values[key] = _int(f"eval.{key}", section[key])
    for key in ("tolerance", "invalid_action_max"):
        if key in section:
            values[key] = _float(f"eval.{key}", section[key])
    threshold_map = {
        "task_success": "min_task_success",
        "tool_precision": "min_tool_precision",
    }
    for toml_key, attr in threshold_map.items():
        if toml_key in thresholds:
            values[attr] = _float(f"thresholds.{toml_key}", thresholds[toml_key])
    return values


def _apply_env(config: EvalConfig, env: Mapping[str, str]) -> EvalConfig:
    updates: dict[str, Any] = {}
    for var, attr in ENV_OVERRIDES.items():
        raw = env.get(var)
        if raw is None or raw == "":
            continue
        if attr in _PATH_ATTRS:
            updates[attr] = Path(raw)
        elif attr in _INT_ATTRS:
            updates[attr] = _int(var, raw)
        elif attr in _FLOAT_ATTRS:
            updates[attr] = _float(var, raw)
        else:
            updates[attr] = raw
    return replace(config, **updates) if updates else config


def load_config(
    path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> EvalConfig:
    """Build the config: defaults <- TOML file <- environment variables.

    ``path`` defaults to ``$AILAB_CONFIG`` or ``eval_config.toml``. An explicitly
    requested file that does not exist is an error; the implicit default is optional.
    """
    env = os.environ if env is None else env
    explicit = path is not None or bool(env.get("AILAB_CONFIG"))
    config_path = Path(path or env.get("AILAB_CONFIG") or DEFAULT_CONFIG_PATH)

    values: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("rb") as handle:
            try:
                values = _from_mapping(tomllib.load(handle))
            except tomllib.TOMLDecodeError as exc:
                raise ConfigError(f"{config_path}: invalid TOML ({exc})") from exc
    elif explicit:
        raise ConfigError(f"config file not found: {config_path}")

    return _apply_env(EvalConfig(**values), env)
