"""Build the README graphics for ailab-agents.

Two structural diagrams (hero, flow) carry no data and are labeled HOW IT WORKS.
Two data graphics (eval anatomy, red-team catalog) are built from committed captures in
captures/, never by hand. Run from the repo root with the vendored kit:

    uv run --with playwright --with pillow python docs/assets/src/render.py docs/assets/src docs/assets
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import readme_kit as k  # noqa: E402

CAPS = HERE / "captures"
HOW = "AILAB-AGENTS . HOW IT WORKS"


def num(pattern: str, text: str) -> str:
    """Pull one captured number out of the real output. Fails loudly if absent."""
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"pattern not found in capture: {pattern!r}")
    return m.group(1)


# -- 1. hero (structural) -----------------------------------------------------------------------

def build_hero() -> str:
    lede = ("A local, dependency-light bounded multi-hop QA agent. It chains a retrieve tool and an "
            "AST-safe calc tool, exposes the same tools over a stdlib MCP server, and gates CI on an "
            "honesty-checked eval. Built on ailab-core.")
    rules = [
        ("Tool use is necessary, not decorative", "every task needs two retrieved facts combined by calc"),
        ("A bounded ReAct loop", "think, then one ACTION tool call or ANSWER, capped at max_steps"),
        ("One registry, two call paths", "the MCP path is byte-identical to the in-process registry"),
        ("The gate fails closed", "a task-naive stub scores 0.0 and cannot pass"),
    ]
    right = k.wheel(["THINK", "RETRIEVE", "CALC", "ANSWER", "GATE"], "AILAB", "AGENT")
    return k.hero(
        kicker="LOCAL AGENT LAB . ZERO DEPENDENCIES",
        title="An agent lab where",
        accent="a number gates CI",
        lede_html=lede,
        rules=rules,
        pill="REACT LOOP . AST-SAFE CALC . MCP PARITY",
        right_html=right,
        footer_left=HOW,
    )


# -- 2. flow (structural) -----------------------------------------------------------------------

def build_flow() -> str:
    y = 290
    h = 168
    xs = [56 + i * 267 for i in range(5)]
    boxes = "".join([
        k.box(xs[0], y, 218, h, "QUESTION + TASK",
              ["16 records, 20 tasks", "required_tools per task", "gold answer + facts"]),
        k.box(xs[1], y, 218, h, "REACT LOOP",
              ["think -> one line", "ACTION tool | ANSWER", "bounded: max_steps 6"]),
        k.box(xs[2], y, 218, h, "TOOL REGISTRY",
              ["retrieve: token overlap", "calc: AST whitelist", "no eval, no exec"]),
        k.box(xs[3], y, 218, h, "TRAJECTORY",
              ["steps + observations", "final answer", "per-task metrics"]),
        k.box(xs[4], y, 218, h, "GATE",
              ["task_success floor", "authenticity + tools", "exit 1 on regression"], accent=True),
    ])
    # The MCP server hangs off the one registry, proving the two call paths cannot drift.
    mcp = k.box(xs[2], y + h + 54, 218, 96, "MCP SERVER",
                ["stdlib JSON-RPC 2.0", "same ToolRegistry"])
    mid = y + h // 2
    specs = [(xs[i] + 218, mid, xs[i + 1], mid) for i in range(4)]
    specs.append((xs[2] + 109, y + h, xs[2] + 109, y + h + 54, "byte-identical", True, "right"))
    sub = "offline, deterministic: StubProvider by default, a recorded gemma3:27b cassette on replay"
    return k.flow(
        kicker="HOW IT WORKS",
        title_html=f"Think, act with a tool, observe, {k.em('then gate')}",
        subline=sub,
        boxes_html=boxes + mcp,
        arrow_specs=specs,
        footer_left=HOW,
    )


# -- 3. eval anatomy (capture-driven) -----------------------------------------------------------

def build_eval() -> str:
    cap = k.load_capture(CAPS / "eval.json")
    out = cap["output"]
    commit = cap["commit"][:7]
    date = num(r"\| (\d{4}-\d{2}-\d{2}) \|", out)
    n = num(r"tasks \(n=(\d+)\)", out)
    verdict = num(r"\| (PASS|FAIL) \|", out)
    ts = num(r"task_success=([\d.]+)", out)
    tp = num(r"tool_precision=([\d.]+)", out)
    inv = num(r"invalid_rate=([\d.]+)", out)
    steps = num(r"mean_steps=([\d.]+)", out)
    auth = re.search(r"provider=(\w+) expected=(\w+) -> (\w+)", out)
    assert auth
    provider, expected, _ = auth.groups()
    calls = num(r"total_tool_calls=(\d+)", out)
    distinct = num(r"distinct_answers=(\d+)", out)
    answered = num(r"answered_n=(\d+)", out)
    real_run = f"AILAB-AGENTS . REAL RUN {date}"

    doc = [
        ("h1", "ailab-eval --provider replay"),
        ("q", f"one real run . commit {k.esc(commit)} . provider {k.esc(provider)} . gemma3:27b . tasks (n={k.esc(n)}) . {k.esc(verdict)}"),
        ("h2", "Primary metric"),
        ("code", f"task_success   {ts}   floor 0.60, band 0.05"),
        ("h2", "Keeping the number honest"),
        ("code", f"tool_precision        {tp}"),
        ("code", f"invalid_action_rate   {inv}"),
        ("code", f"mean_steps            {steps}"),
        ("h2", "Authenticity and task effect"),
        ("code", f"provider {provider}  expected {expected}  authentic"),
        ("code", f"tool calls {calls} across {n} tasks"),
        ("code", f"distinct answers {distinct} of {answered}"),
        ("m", "generated from eval_results.json, never typed by hand"),
    ]
    notes = [
        (150, f"task_success {ts} is 18 of 20, not 1.0: two tasks are still missed, so the gate can still catch a regression."),
        (300, f"tool_precision {tp} and invalid_action_rate {inv}: the model called only the required tools and emitted no malformed step."),
        (430, f"provider is {provider}, not stub: a stub or sabotage run fails the authenticity check and cannot reach a green gate."),
        (520, f"distinct answers {distinct} of {answered}: every task got its own answer, so a single fixed-string reply cannot pass."),
    ]
    return k.anatomy("ANATOMY OF A REAL EVAL", doc, notes, real_run)


# -- 4. red-team gate catalog (capture-driven) --------------------------------------------------

def build_red() -> str:
    cap = k.load_capture(CAPS / "red.json")
    out = cap["output"]
    date = cap["captured_at"][:10]
    real_run = f"AILAB-AGENTS . REAL RUN {date}"
    cases = re.findall(r"PASS \[(\S+) ([^\]]+)\]: exit (\d+), named '([^']+)'", out)
    if len(cases) != 6:
        raise SystemExit(f"expected 6 red cases in capture, found {len(cases)}")
    detail = {
        "task_success": "task_success 0.00 < floor 0.60",
        "tool_usage": "no tool was called",
        "non_triviality": "one answer for every task",
        "invalid_action_rate": "every step malformed",
        "empty": "task set is empty",
        "min_tasks": "fewer tasks than min_tasks 8",
    }
    cards = []
    for cid, label, code, metric in cases:
        dim = detail.get(metric, f"named {metric}")
        cards.append((f"SABOTAGE {cid}", label, f"exit {code} . {metric}", dim))
    sub = ("make red . six sabotages across four gate checks and two config guards . "
           "each exits non-zero and names the metric it tripped")
    return k.catalog(
        "THE GATE MUST FAIL ON PURPOSE",
        f"A gate that {k.em('cannot fail')} proves nothing",
        sub,
        cards,
        real_run,
        cols=6,
        card_height=210,
    )


if __name__ == "__main__":
    pages = {
        "hero": build_hero(),
        "architecture": build_flow(),
        "anatomy": build_eval(),
        "red-gate": build_red(),
    }
    k.write_pages(HERE, pages)
