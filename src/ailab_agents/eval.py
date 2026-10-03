"""Agent eval runner and regression gate.

Usage::

    ailab-eval                                   # or: python -m ailab_agents
    ailab-eval --provider replay --cassette fixtures/cassettes/agent.json
    ailab-eval --min-task-success 0.7 --tolerance 0.05

The gate uses a TOLERANCE BAND, not an exact threshold, and layers honesty checks ON TOP of the
numeric floors: the provider must be the recorded real model (a stub cannot pass), the agent must
actually call tools, and the per-task answers must not be byte-identical (an agent that ignores
the task cannot pass). Exit codes: 0 = pass, 1 = regression (a metric below ``floor - tolerance``,
no tool use, no task effect, too many invalid actions, or a non-authentic provider), 2 = bad
config or data (empty corpus/tasks, or fewer than ``min_tasks`` tasks).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ailab_core.gate import floor_breach, require_metrics

from ailab_agents.agent import AgentConfig, Provider, Trajectory, run_agent
from ailab_agents.config import ConfigError, EvalConfig, load_config
from ailab_agents.data import DatasetError, Task, load_corpus, load_tasks, sha256_file
from ailab_agents.metrics import (
    invalid_action_rate,
    mean,
    task_success,
    tool_precision,
    trajectory_steps,
)
from ailab_agents.providers import CassetteMissError, ReplayProvider, StubProvider
from ailab_agents.tools import build_registry

SCHEMA_VERSION = 1

#: fixed outputs for the proof-of-gate sabotages (see scripts/prove_gate.sh).
HARDCODED_ANSWER = "FIXED-ANSWER"
INVALID_OUTPUT = "I am not sure, let me think about it some more."

TABLE_HEADER = (
    "| Date | Commit | Provider | Dataset | max_steps | task_success | tool_precision "
    "| invalid_rate | mean_steps | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)


class _HardcodeProvider:
    """Proof-of-gate: answer every task with one fixed string (trips the non-triviality check)."""

    name = "stub"
    model = "stub-agent-v1"

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def complete(self, prompt: str) -> str:
        return f"ANSWER: {self._answer}"


class _InvalidProvider:
    """Proof-of-gate: emit a malformed line every step (trips invalid_action_rate)."""

    name = "stub"
    model = "stub-agent-v1"

    def complete(self, prompt: str) -> str:
        return INVALID_OUTPUT


def current_commit() -> str | None:
    """Commit for the results record: env first (CI, Docker), then git, else None."""
    for var in ("GITHUB_SHA", "AILAB_COMMIT"):
        if sha := os.environ.get(var):
            return sha if sha != "unknown" else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def run_tasks(
    tasks: Sequence[Task],
    provider: Provider,
    config: EvalConfig,
    *,
    no_tools: bool = False,
) -> list[Trajectory]:
    """Run the bounded agent over every task and return the trajectories."""
    records = load_corpus(config.corpus)
    registry = build_registry(records)
    agent_config = AgentConfig(max_steps=config.max_steps, no_tools=no_tools)
    return [run_agent(task.id, task.question, provider, registry, agent_config) for task in tasks]


def score(tasks: Sequence[Task], trajectories: Sequence[Trajectory]) -> dict[str, Any]:
    """Aggregate per-task metrics into the schema-1 metric dict plus diagnostics."""
    paired = list(zip(trajectories, tasks, strict=True))
    successes = [task_success(traj, task.answer) for traj, task in paired]
    precisions = [tool_precision(traj, task.required_tools) for traj, task in paired]
    precisions_present = [p for p in precisions if p is not None]
    invalid_rates = [invalid_action_rate(traj) for traj in trajectories]
    step_counts = [float(trajectory_steps(traj)) for traj in trajectories]
    total_tool_calls = sum(len(traj.tool_calls) for traj in trajectories)
    answers = [traj.final_answer for traj in trajectories if traj.final_answer is not None]
    return {
        "task_success": mean(successes),
        "tool_precision": mean(precisions_present),
        "invalid_action_rate": mean(invalid_rates),
        "mean_steps": mean(step_counts),
        "n": len(tasks),
        "total_tool_calls": total_tool_calls,
        "distinct_answers": len(set(answers)),
        "answered_n": len(answers),
    }


def evaluate(
    config: EvalConfig,
    provider: Provider | None = None,
    *,
    no_tools: bool = False,
) -> dict[str, Any]:
    """Run one eval and return the results record written to ``eval_results.json``.

    Raises :class:`DatasetError` for empty data or a task set smaller than ``min_tasks`` (exit 2).
    """
    tasks = load_tasks(config.tasks)
    if len(tasks) < config.min_tasks:
        raise DatasetError(
            f"{config.tasks}: {len(tasks)} tasks is below min_tasks {config.min_tasks}"
        )
    active: Provider = provider or StubProvider()
    trajectories = run_tasks(tasks, active, config, no_tools=no_tools)
    agg = score(tasks, trajectories)

    metrics = {
        "task_success": _round(agg["task_success"]),
        "tool_precision": _round(agg["tool_precision"]),
        "invalid_action_rate": _round(agg["invalid_action_rate"]),
        "mean_steps": _round(agg["mean_steps"]),
        "n": agg["n"],
    }
    failures = _gate(config, metrics, agg, active)
    return {
        "schema_version": SCHEMA_VERSION,
        "lab": config.lab,
        "dataset": config.tasks.stem,
        "provider": active.name,
        "model": active.model,
        "primary_metric": config.primary_metric_key,
        "metrics": metrics,
        "threshold": config.thresholds[config.primary_metric],
        "passed": not failures,
        "commit": current_commit(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        # Diagnostic extras below (not part of the schema_version 1 contract).
        "corpus_sha256": sha256_file(config.corpus),
        "tasks_sha256": sha256_file(config.tasks),
        "config": {
            "max_steps": config.max_steps,
            "provider": config.provider,
            "min_tasks": config.min_tasks,
            "tolerance": config.tolerance,
            "invalid_action_max": config.invalid_action_max,
        },
        "tolerance": config.tolerance,
        "thresholds": config.thresholds,
        "authenticity": {
            "provider": active.name,
            "expected": config.real_provider,
            "authentic": active.name == config.real_provider,
        },
        "tool_usage": {
            "total_tool_calls": agg["total_tool_calls"],
            "no_tools_mode": no_tools,
        },
        "non_triviality": {
            "distinct_answers": agg["distinct_answers"],
            "answered_n": agg["answered_n"],
        },
        "failures": failures,
    }


def _gate(
    config: EvalConfig,
    metrics: dict[str, Any],
    agg: dict[str, Any],
    provider: Provider,
) -> list[str]:
    """Build the list of regression messages; empty means the gate passes."""
    tol = config.tolerance
    failures: list[str] = []
    failures.extend(require_metrics(metrics, [config.primary_metric]))
    for base, floor in config.thresholds.items():
        if msg := floor_breach(base, metrics.get(base), floor, tol):
            failures.append(msg)
    if provider.name != config.real_provider:
        failures.append(
            f"authenticity: provider {provider.name!r} is not the recorded real model "
            f"{config.real_provider!r}; a stub or sabotage run cannot pass the gate"
        )
    if agg["total_tool_calls"] <= 0:
        failures.append("tool_usage: the agent used no tools across the run (tool use is required)")
    if agg["n"] >= 2 and agg["answered_n"] >= 2 and agg["distinct_answers"] == 1:
        failures.append(
            "non_triviality: agent output shows no task effect "
            "(final answers byte-identical across distinct tasks)"
        )
    rate = metrics["invalid_action_rate"]
    if rate is not None and rate > config.invalid_action_max:
        failures.append(
            f"invalid_action_rate {rate:.4f} > max {config.invalid_action_max:.4f}; "
            "the agent emitted malformed actions"
        )
    return failures


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def markdown_rows(result: dict[str, Any]) -> str:
    """README "Eval results" row, derived only from a results record."""
    commit = (result["commit"] or "unknown")[:7]
    cfg = result["config"]
    dataset = f"{result['dataset']} (n={result['metrics']['n']})"
    note = "PASS" if result["passed"] else "FAIL"
    m = result["metrics"]
    return (
        f"| {result['generated_at'][:10]} | {commit} | {result['provider']} | {dataset} "
        f"| {cfg['max_steps']} | {_fmt(m['task_success'])} | {_fmt(m['tool_precision'])} "
        f"| {_fmt(m['invalid_action_rate'])} | {_fmt(m['mean_steps'])} | {note} |"
    )


def _print_summary(result: dict[str, Any]) -> None:
    """Human-readable gate summary printed after the table."""
    auth = result["authenticity"]
    print(
        f"authenticity: provider={auth['provider']} expected={auth['expected']} "
        f"-> {'ok' if auth['authentic'] else 'NOT AUTHENTIC'}"
    )
    usage = result["tool_usage"]
    print(
        f"tool usage: total_tool_calls={usage['total_tool_calls']} "
        f"no_tools_mode={usage['no_tools_mode']}"
    )
    nt = result["non_triviality"]
    print(
        f"non-triviality: distinct_answers={nt['distinct_answers']} answered_n={nt['answered_n']}"
    )
    m = result["metrics"]
    print(
        f"metrics: task_success={_fmt(m['task_success'])} "
        f"tool_precision={_fmt(m['tool_precision'])} "
        f"invalid_rate={_fmt(m['invalid_action_rate'])} mean_steps={_fmt(m['mean_steps'])} "
        f"(n={m['n']})"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ailab-eval", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="TOML config path ($AILAB_CONFIG or eval_config.toml)")
    parser.add_argument("--corpus", type=Path, help="override the corpus JSONL")
    parser.add_argument("--tasks", type=Path, help="override the tasks JSONL")
    parser.add_argument("--max-steps", type=int, dest="max_steps", help="override max_steps")
    parser.add_argument("--output", type=Path, help="override the results JSON path")
    parser.add_argument(
        "--min-task-success",
        type=float,
        dest="min_task_success",
        help="override the task_success floor",
    )
    parser.add_argument("--tolerance", type=float, help="override the gate tolerance band")
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="proof-of-gate: hide every tool so the agent answers blind",
    )
    parser.add_argument(
        "--hardcode-answer",
        action="store_true",
        help="proof-of-gate: answer every task with one fixed string",
    )
    parser.add_argument(
        "--force-invalid",
        action="store_true",
        help="proof-of-gate: emit a malformed action every step",
    )
    parser.add_argument(
        "--provider",
        choices=["stub", "replay"],
        default="stub",
        help="agent brain: offline stub (default) or replay a recorded cassette",
    )
    parser.add_argument(
        "--cassette",
        type=Path,
        default=Path("fixtures/cassettes/agent.json"),
        help="cassette path for --provider replay",
    )
    parser.add_argument(
        "--model",
        default="gemma3:27b",
        help="recorded model id to replay (keys are sha256(model, prompt))",
    )
    parser.add_argument(
        "--table-from",
        type=Path,
        metavar="RESULTS_JSON",
        help="print a README table row from an existing results file, without running",
    )
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    keys = ("corpus", "tasks", "max_steps", "output", "min_task_success", "tolerance")
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def _build_provider(args: argparse.Namespace, config: EvalConfig) -> Provider:
    if args.provider == "replay":
        return ReplayProvider(args.cassette, name=config.real_provider, model=args.model)
    if args.hardcode_answer:
        return _HardcodeProvider(HARDCODED_ANSWER)
    if args.force_invalid:
        return _InvalidProvider()
    return StubProvider()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.table_from is not None:
        try:
            record = json.loads(args.table_from.read_text(encoding="utf-8"))
            print(f"{TABLE_HEADER}\n{markdown_rows(record)}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"ailab-eval: error: cannot read {args.table_from}: {exc!r}", file=sys.stderr)
            return 2
        return 0
    try:
        config = replace(load_config(args.config), **_overrides(args))
        provider = _build_provider(args, config)
        result = evaluate(config, provider, no_tools=args.no_tools)
    except (ConfigError, DatasetError, OSError, CassetteMissError) as exc:
        print(f"ailab-eval: error: {exc}", file=sys.stderr)
        return 2

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    table = f"{TABLE_HEADER}\n{markdown_rows(result)}"
    print(table)
    _print_summary(result)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"### Eval results\n\n{table}\n\n")

    if not result["passed"]:
        for failure in result["failures"]:
            print(f"REGRESSION: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
