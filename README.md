# ailab-agents

A small, local, open-source lab for a bounded, tool-using agent, where **a number gates CI**. It
builds a ReAct-style agent that MUST chain a `retrieve` tool and a `calc` tool to answer, scores
whether it got the answer AND used the tools, and a tolerance-band regression gate fails the build
when the agent regresses. The same tools are exposed over a tiny MCP server, and a parity test
proves the two call paths cannot drift. You learn the concepts by running them: change one input,
re-run, and watch the trajectory and the metric move.

- **Tool use is necessary, not decorative.** Every multi-hop task is answerable only by retrieving two facts and combining them with `calc`. An agent that does not chain the tools cannot pass.
- **A gate you can trust because you watched it fail.** `make red` fires six sabotages and each one must exit non-zero and name the metric it tripped.
- **Honest checks.** On top of the numeric floors the gate also checks provider authenticity (a stub is not the real model), tool usage, and non-triviality (answers must differ by task).
- **One registry, two call paths.** The agent and the MCP server share one `ToolRegistry`, and a parity test asserts the MCP results are byte-identical to the in-process calls.
- **Safe calc.** The calc tool uses an AST whitelist, never `eval`/`exec`. A red test proves `__import__("os")` and attribute-access payloads raise.
- **Offline and reproducible.** A stub brain by default, a recorded model on replay. No network and no key to run the eval.

> **The one idea worth stealing, even if you never run this code:** a passing score is not a
> passing agent. This lab's gate refuses a run that scores by luck, that echoes one fixed answer,
> or that never calls a tool, and it refuses any provider that is not the recorded real model. A
> green gate here means the agent did the work, with the tools, as itself, and you can prove the
> gate goes red when any of that is false.

---

## What it measures

The primary metric is `task_success`: did the agent's final answer match the gold answer, after
normalisation. Three more metrics and three honesty checks keep that number honest:

| Metric or check | What it catches |
|---|---|
| `task_success` | the agent did not reach the right answer |
| `tool_precision` | the agent used tools outside the task's required set |
| `invalid_action_rate` | the agent emitted malformed, unparseable steps |
| `mean_steps` | a degenerate loop (surfaced, not gated) |
| authenticity | the provider is a stub or sabotage, not the recorded real model |
| tool usage | the agent answered without calling any tool |
| non-triviality | the agent returned one fixed answer regardless of the task |

## The agent and its tools

The agent is a bounded ReAct loop (`max_steps` 6). Each step it sees the question, the tool list
and the observation history, then emits one line: `ACTION: <tool> <arg>` to call a tool, or
`ANSWER: <text>` to stop. A malformed line is an invalid action, counted and bounded.

- `retrieve`: a deterministic token-overlap lookup over the corpus. Input a query, get the best
  record's text. No index, no network.
- `calc`: arithmetic over `+ - * /` and parentheses, parsed with an AST whitelist and evaluated by
  walking the tree. Names, calls, attributes, subscripts, comprehensions, the power operator and
  dunders all raise. There is no python `eval`/`exec` in the tool.

## 60 seconds to a scored eval

```bash
make install    # uv sync --locked (Python 3.12, dev tools included)
make eval       # score the agent, write eval_results.json, enforce the gate
```

By default `make eval` runs the offline STUB brain, which is task-naive on purpose: it retrieves
once and answers with the raw text instead of doing the arithmetic. So it FAILS the gate, and
that is the point. This is the real output:

```text
| Date | Commit | Provider | Dataset | max_steps | task_success | tool_precision | invalid_rate | mean_steps | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | unknown | stub | tasks (n=20) | 6 | 0.0000 | 1.0000 | 0.0000 | 2.0000 | FAIL |
authenticity: provider=stub expected=ollama -> NOT AUTHENTIC
tool usage: total_tool_calls=20 no_tools_mode=False
non-triviality: distinct_answers=11 answered_n=20
metrics: task_success=0.0000 tool_precision=1.0000 invalid_rate=0.0000 mean_steps=2.0000 (n=20)
```

The stub opens the right drawer (`tool_precision` 1.0) and still gets the number wrong
(`task_success` 0.0). A green gate requires a real recorded model that chains the tools; see
[Recording the cassette](#recording-the-cassette).

Replaying the committed real `gemma3:27b` cassette passes the gate. This is the committed
`eval_results.json`:

```text
| Date | Commit | Provider | Dataset | max_steps | task_success | tool_precision | invalid_rate | mean_steps | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | unknown | ollama | tasks (n=20) | 6 | 0.9000 | 1.0000 | 0.0000 | 2.7500 | PASS |
```

The real model chains retrieve then calc and reaches the right answer on 18 of 20 tasks
(`task_success` 0.90). It is not saturated: two tasks are still missed, so the gate can detect a
regression. Reproduce it offline with
`ailab-eval --provider replay --cassette fixtures/cassettes/agent_gemma3_27b.json`.

`make demo` shows the agent's steps on a couple of tasks first:

```text
== ailab-agents demo ==

corpus: 16 records, 20 tasks, tools=['retrieve', 'calc'], max_steps=6

Task [t01]: What is the combined crew of Halcyon Ridge Station and Vesper Hollow Station?
    required_tools=['retrieve', 'calc'] gold=79
    step 1: ACTION retrieve What is the combined crew of Halcyon Ridge ...
            -> Halcyon Ridge Station was commissioned in 2131. It carries a crew of 42 and operates 18 ...
    step 2: ANSWER Halcyon Ridge Station was commissioned in 2131. It carries a crew of 42 ...
    final: '...' -> MISS (stopped=answered)
```

### All the targets

```bash
make lint       # ruff check, ruff format --check, mypy --strict
make test       # pytest with branch coverage (fails under 90%)
make red        # prove the gate FAILS on sabotages, each naming its metric
make mcp        # list the tools over a real MCP stdio round-trip
make compare    # diff two saved runs that change ONE variable
make record     # record the real-model cassette (needs OLLAMA_HOST; opt-in)
make lint-docs  # fail on an em dash (U+2014) in docs/
make check-learning-numbers  # assert docs/LEARNING.md figures match the results
make scan       # gitleaks over full history + denylist scan

docker compose up   # build the image and run the demo in a container
```

## Configure it

Configuration is layered: defaults, then `eval_config.toml`, then environment variables, then CLI
flags. `ailab-eval` exits `0` when the gate passes, `1` on a regression (a metric outside its band,
no tool use, no task effect, too many invalid actions, or a non-authentic provider), and `2` on a
bad config or dataset (empty tasks, or fewer than `min_tasks`).

| Variable | Default | Purpose |
|---|---|---|
| `AILAB_CORPUS` | `fixtures/corpus.jsonl` | Corpus JSONL (`{"id","text"}`) |
| `AILAB_TASKS` | `fixtures/tasks.jsonl` | Tasks JSONL (`{"id","question","answer","required_tools","gold_facts"}`) |
| `AILAB_PROVIDER` | `stub` | `stub` or `replay` |
| `AILAB_MODEL` | `gemma3:27b` | Recorded model id to replay |
| `AILAB_MAX_STEPS` | `6` | Bounded loop: steps per task |
| `AILAB_TOLERANCE` | `0.05` | Gate band below each floor |
| `AILAB_INVALID_ACTION_MAX` | `0.20` | Max fraction of malformed steps |
| `AILAB_MIN_TASKS` | `8` | Fail (exit 2) below this many tasks |
| `AILAB_OUTPUT` | `eval_results.json` | Results file |

### Results file contract

`eval_results.json` keeps the template's versioned shape (`schema_version: 1`). These top-level
keys are always present; the agent fields (authenticity, tool_usage, non_triviality, config,
shas) are diagnostic extras.

```json
{
  "schema_version": 1,
  "lab": "ailab-agents",
  "dataset": "tasks",
  "provider": "stub",
  "model": "stub-agent-v1",
  "primary_metric": "task_success",
  "metrics": {"task_success": 0.0, "tool_precision": 1.0, "invalid_action_rate": 0.0, "mean_steps": 2.0, "n": 20},
  "threshold": 0.6,
  "passed": false,
  "commit": "<full git sha, or null>",
  "generated_at": "2026-10-03T00:00:00Z"
}
```

## How it works

The corpus and the tasks are loaded, a `ToolRegistry` is built (a `retrieve` tool over the corpus
plus the `calc` tool), and the bounded agent runs each task. Per-task metrics are aggregated, and
the gate checks them against explicit floors with a tolerance band, plus the authenticity,
tool-usage and non-triviality checks. A regression exits `1` and CI fails.

The brain runs behind an `LLMProvider` seam imported from `ailab-core`. `StubProvider` is a
deterministic, offline format path, not a real model. `ReplayProvider` serves a committed JSON
cassette keyed by `sha256(model, prompt)`, recorded once from a real `gemma3:27b` (`make record`,
opt-in). A cassette miss raises, so a stale cassette cannot silently pass. Nothing in the default
path touches the network.

| Path | What lives there |
|---|---|
| `src/ailab_agents/data.py` | Corpus and task loaders, dataset sha256 |
| `src/ailab_agents/tools.py` | `Tool` protocol, `RetrieveTool`, `CalcTool` (AST whitelist), `ToolRegistry` |
| `src/ailab_agents/agent.py` | Bounded ReAct loop, action parser, `Trajectory` |
| `src/ailab_agents/metrics.py` | task_success, tool_precision, invalid_action_rate, normalize_answer |
| `src/ailab_agents/providers.py` | `LLMProvider`, `StubProvider`, `ReplayProvider`, `OllamaProvider` |
| `src/ailab_agents/eval.py` | Runner and regression gate (`ailab-eval`) |
| `src/ailab_agents/mcp_server.py` | Stdlib JSON-RPC MCP server over the same registry |
| `src/ailab_agents/compare.py` | Diff two saved runs (one changed variable) |
| `fixtures/` | Synthetic data only; provenance in [FIXTURES.md](FIXTURES.md) |

It keeps **no third-party runtime dependencies**: the tools and the MCP server are pure stdlib,
and the live provider uses `urllib`. It has one pinned first-party dependency, `ailab-core`,
referenced by an immutable commit SHA for the provider seam and the gate primitives. `ruff`,
`mypy` and `pytest` are dev-only. It is scaffolded from `ailab-template-python` and ports its
wiring from `ailab-rag` (SHAs in [`BUILD_SOURCES.txt`](BUILD_SOURCES.txt) and
[docs/DESIGN.md](docs/DESIGN.md)).

## Prove the gate can fail

A passing gate is only worth something if it could have failed. `make red` breaks the agent six
ways on purpose, and each one must exit non-zero and name the metric it tripped. If any sabotage
slips through, `make red` itself fails.

```text
== make red: proving the gate fails on each sabotage ==
PASS [1 stub below task_success floor]: exit 1, named 'task_success'
PASS [2 no tools (answers blind)]: exit 1, named 'tool_usage'
PASS [3 hardcoded answer]: exit 1, named 'non_triviality'
PASS [4 forced invalid actions]: exit 1, named 'invalid_action_rate'
PASS [5 empty tasks (exit 2)]: exit 2, named 'empty'
PASS [6 below min_tasks (exit 2)]: exit 2, named 'min_tasks'
== all red cases failed the gate as required ==
```

## The MCP server

The same tools are exposed over a tiny stdlib JSON-RPC 2.0 server on stdio, with three methods:
`initialize`, `tools/list`, `tools/call`. It reads one JSON request per line and writes one JSON
response per line. No resources, prompts, notifications or network.

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"calc","arguments":{"input":"42+37"}}}' \
  | uv run python -m ailab_agents.mcp_server
# -> {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "79"}]}}
```

`tests/test_mcp_parity.py` proves the MCP path returns results byte-identical to the in-process
`ToolRegistry.call(...)` for every tool across every task argument, because both paths dispatch
through the one registry.

## Recording the cassette

The default stub cannot pass the gate. A green eval needs a real recorded model:

```bash
OLLAMA_HOST=http://<host>:11434 AILAB_RECORD_MODEL=gemma3:27b make record
uv run ailab-eval --provider replay --cassette fixtures/cassettes/agent_gemma3_27b.json --model gemma3:27b
```

`make record` drives the SAME bounded loop with a local Ollama model and writes a cassette keyed
by `sha256(model, prompt)`. The host address comes from `OLLAMA_HOST` and is never written to
disk; a scan refuses to write if any host, IP or local path leaked into a response. This step is
operator-gated and is not part of v1's committed data.

## Scope: what it does not do

- **The committed cassette is a stub, not a real model.** `fixtures/cassettes/agent.json` is a
  format recording for offline tests; its header says `provider: stub`, so replaying it can never
  pass the authenticity gate.
- **The stub brain does not reason.** It retrieves once and echoes the text. A real model is
  needed to chain retrieve then calc.
- **No live LLM in CI.** The real model is recorded once and replayed. A live call is opt-in and
  local.
- **The data is synthetic.** The 16 research stations are invented. The numbers describe this lab,
  not any real agent.
- **Two tools, one loop.** No planner, no reflection, no parallel calls, no multi-agent
  orchestration. Those are later slices behind the seam.

## Privacy and the secret wall

The data is synthetic (invented research stations) and carries no real product, support, or
personal data. The secret wall runs on every push:

1. **gitleaks** with [`.gitleaks.toml`](.gitleaks.toml), run by the `pre-push` hook (`make hooks`)
   and the CI `secret-scan` job, over the full history.
2. **Denylist scan** ([`scripts/denylist_scan.sh`](scripts/denylist_scan.sh)): generic public
   patterns are committed; private patterns come from a secret or local file and are never
   committed or printed. The scan fails closed.

`OLLAMA_HOST` is runtime-only, read from the environment, and never written to a committed file.

## License

[Apache License 2.0](LICENSE). Copyright 2026 Andrew Pyle.
