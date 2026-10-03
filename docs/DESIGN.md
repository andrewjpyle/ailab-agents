# Design notes

Decisions behind `ailab-agents`, and the alternatives that were considered and rejected. This
lab replaces the retrieve-then-read domain of its sibling with a bounded, tool-using agent,
and keeps the template's discipline: a number gates CI, and the gate can be proven to fail.

## Provenance

This lab was scaffolded from three sources at `origin/main`:

- `ailab-template-python` at `f17855c`: the eval runner shape, the regression gate, config
  layering, the secret wall (pre-push hook plus CI), non-root Docker, ruff, mypy, pytest with
  a coverage floor, and the `eval_results.json` `schema_version` contract.
- `ailab-rag` at `2a2c230`: the eval and gate wiring, the config and data loaders, the
  record-and-replay provider pattern and the proof-of-gate script, ported and then rewritten
  for the agents domain.
- `ailab-core` at `06856ce`: imported, not copied. The provider seam
  (`ailab_core.providers`) and the gate primitives (`ailab_core.gate`) are a real dependency
  pinned by commit SHA.

The SHAs are also recorded in `BUILD_SOURCES.txt` and `FIXTURES.md`.

## 1. The seam is a bounded ReAct loop plus a tool Protocol

The lab's unit of study is an agent that MUST chain tools to answer. A `Tool` is anything with
a `name`, a `description` and a `call(arg) -> str`. The agent lists the tools, reads the
observation history, and emits one line per step: `ACTION: <tool> <arg>` or `ANSWER: <text>`.
The loop is bounded by `max_steps`.

- Rejected: a free-form agent with open-ended output. A tight action grammar keeps the parser,
  the dispatch and the metrics auditable, and makes a malformed step a measurable event rather
  than a crash.
- Rejected: an agent framework. Heavy dependency trees and a vendor choice every lab would
  inherit. This lab keeps its runtime dependency list to the single pinned `ailab-core`.

## 2. Tool use is necessary, not decorative

Every multi-hop task is answerable ONLY by retrieving two or more facts and combining them with
`calc`. A model that does not chain the tools cannot get the number right. This is what makes
`task_success` a real measure of agentic behaviour rather than of recall from the prompt.

- Rejected: tasks answerable from the question alone. Then a non-tool-using model could pass,
  and the eval would measure nothing about tool use. The single-hop tasks (retrieve only) are
  kept as a contrast, not as the bar.

## 3. The calc tool uses an AST whitelist, never `eval`

`CalcTool` parses the expression with `ast.parse(mode="eval")` and walks the tree, allowing only
numeric literals and the binary operators `+ - * /` plus unary minus, under a length cap. A name,
a call, an attribute, a subscript, a comprehension, the power operator and a dunder all raise.
There is NO python `eval`/`exec` anywhere in the tool.

- Rejected: `eval()` with a restricted namespace. Sandboxing `eval` by clearing builtins is a
  known trap: attribute walks from a literal can still reach dangerous objects. A whitelist that
  never executes arbitrary nodes is the safe default, and `tests/test_calc_safety.py` proves the
  unsafe payloads raise.

## 4. The registry is the one source both the agent and MCP read

`ToolRegistry` holds the tools and is passed to BOTH the in-process agent loop AND the MCP
server. A tool therefore has exactly one implementation, and the two call paths cannot drift.

- Rejected: a second tool implementation inside the MCP server. That is the drift this lab is
  built to prevent; `tests/test_mcp_parity.py` asserts the MCP path returns results
  byte-identical to `registry.call(...)` for every tool across every task argument.

## 5. The gate layers honesty checks on the numeric floors

The gate fails below `floor - tolerance`, not on an exact score, so it does not trip on noise.
On top of the `task_success` and `tool_precision` floors it also fails on:

- provider authenticity: a stub or sabotage provider is not the recorded real model, so it
  cannot pass. A green gate requires a run whose provider name is the recorded model (`ollama`).
- tool usage: the run must make at least one tool call, or it reports "the agent used no tools".
- non-triviality: the per-task final answers must not be byte-identical across distinct tasks,
  or it reports "agent output shows no task effect".
- invalid actions: the fraction of malformed steps must stay at or below `invalid_action_max`.

- Rejected: an exact threshold. A single point score flaps on a small set.
- Rejected: trusting `task_success` alone. A provider could score by luck, by echoing a fixed
  string, or by never using a tool. The honesty checks name each of those failures.

## 6. The stub is a format path, not a real pass

`StubProvider` is a deterministic, offline agent brain that is FORMAT-valid but TASK-naive: it
emits well-formed `ACTION:`/`ANSWER:` lines, so the loop, parser, dispatch and metrics are all
exercised with no network, but it does one retrieve and then answers with the raw retrieved text,
never doing the arithmetic. So `task_success` is low with the stub, and the gate fails. This is
deliberate: the stub proves the machinery, and a real recorded model proves the task.

- Rejected: a stub that passes the gate. That would let the lab claim a green result with no real
  model, which is exactly the overclaim this template exists to prevent.

## 7. The live provider is record-and-replay, not live in CI

`OllamaProvider` records real completions once (via `make record`), driving the SAME bounded
loop the eval runs. `ReplayProvider` serves a committed JSON cassette keyed by
`sha256(model, prompt)`; a miss raises, so a stale cassette cannot silently pass. The committed
cassette (`fixtures/cassettes/agent.json`) is a STUB recording for offline tests, clearly
labelled in its header (`provider: stub`). The real `gemma3:27b` cassette is recorded separately
by the operator-gated step and lives at `fixtures/cassettes/agent_gemma3_27b.json`.

- Rejected: a live model in CI. Non-deterministic, costs money or needs a server, and turns an
  outage into a red build. Recording once and replaying keeps the real model's behaviour without
  the flakiness.
- Rejected: committing a stub recording labelled as the real model. That would fake the pass the
  lab is built to earn honestly.

## 8. The MCP server is hand-rolled stdlib JSON-RPC

A minimal JSON-RPC 2.0 server over stdio exposes `initialize`, `tools/list` and `tools/call`
only. No resources, prompts, notifications or network. `tools/call` returns
`{"content":[{"type":"text","text": <result>}]}`, where the text is exactly
`registry.call(name, arg)`.

- Rejected: an MCP SDK dependency. The lab is stdlib-only at runtime, and a hand-rolled server of
  three methods is small enough to audit and keeps the parity contract obvious.

## 9. Kept from the template

The secret wall (pre-push hook plus a required CI check, both scanning full history), the
non-root multi-stage Docker image, uv with a committed lockfile, strict mypy, a 90 percent
coverage floor, and the versioned `eval_results.json` contract. These are not re-argued here.

## 10. Content-rule enforcement

`make lint-docs` fails on the em dash character (U+2014) anywhere in the docs. `make
check-learning-numbers` fails if a tagged figure in `docs/LEARNING.md` differs from the committed
results. Both run in CI, so a hand-typed or drifted number cannot merge. Working tree verified
clean before hand-off.

## 11. Things deliberately left out of v1 (the stop-list)

- A real-model pass in CI. The green eval needs the operator-recorded `gemma3:27b` cassette.
- More than two tools, parallel tool calls, or a planner or reflection step.
- Multi-agent orchestration, memory across tasks, or a browser UI.
- MCP resources, prompts, notifications, or any network transport.
- README graphics (the webp kit). A documented follow-up; v1 is text-first.

Each is a later slice behind the seam this lab already defines.
