"""Compare two eval result files that change exactly ONE config variable.

Usage::

    python -m ailab_agents.compare results/run_tools.json results/run_notools.json

It prints the single changed variable and the delta on each shared metric. It refuses to
compare two runs that differ in more than one config variable, because then the delta could
not be attributed to one cause.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def changed_variables(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Return the config keys whose value differs between two result records."""
    cfg_a, cfg_b = a["config"], b["config"]
    keys = sorted(set(cfg_a) | set(cfg_b))
    return {k: (cfg_a.get(k), cfg_b.get(k)) for k in keys if cfg_a.get(k) != cfg_b.get(k)}


def metric_deltas(a: dict[str, Any], b: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    """Return ``(metric, a_value, b_value, delta)`` for each shared numeric metric."""
    ma, mb = a["metrics"], b["metrics"]
    rows: list[tuple[str, float, float, float]] = []
    for key in ma:
        if key == "n" or key not in mb:
            continue
        va, vb = ma[key], mb[key]
        if isinstance(va, int | float) and isinstance(vb, int | float):
            rows.append((key, float(va), float(vb), float(vb) - float(va)))
    return rows


def render(a: dict[str, Any], b: dict[str, Any]) -> str:
    changed = changed_variables(a, b)
    lines = [
        f"A: provider={a['provider']} max_steps={a['config']['max_steps']} (passed={a['passed']})",
        f"B: provider={b['provider']} max_steps={b['config']['max_steps']} (passed={b['passed']})",
    ]
    if len(changed) != 1:
        lines.append(f"ERROR: runs differ in {len(changed)} variables ({sorted(changed)}); need 1.")
        return "\n".join(lines)
    name, (va, vb) = next(iter(changed.items()))
    lines.append(f"\nsingle changed variable: {name}: {va} -> {vb}\n")
    lines.append(f"{'metric':<22} {'A':>10} {'B':>10} {'delta':>10}")
    for metric, av, bv, delta in metric_deltas(a, b):
        lines.append(f"{metric:<22} {av:>10.4f} {bv:>10.4f} {delta:>+10.4f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m ailab_agents.compare RUN_A.json RUN_B.json", file=sys.stderr)
        return 2
    a = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    b = json.loads(Path(args[1]).read_text(encoding="utf-8"))
    rendered = render(a, b)
    print(rendered)
    return 1 if "ERROR" in rendered else 0


if __name__ == "__main__":
    raise SystemExit(main())
