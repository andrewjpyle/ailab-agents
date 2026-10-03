# Learning notes

This lab teaches bounded, tool-using agents by running. Every tagged number below is generated
by a committed results file, never typed by hand. A CI check (`make check-learning-numbers`)
fails if a tagged figure drifts from the results.

Two kinds of statement appear here. "Observed in this lab" states a number this lab produced. A
"[VERIFY]" claim is general and is not proven by this lab's data.

Honesty note up front: the tagged numbers here come from the STUB baseline
(`results/stub_baseline.json`), the task-naive floor. The stub is a FORMAT path, not a real
model, so its `task_success` is 0.0 <!--lab:results/stub_baseline.json:metrics.task_success-->
on the 20 <!--lab:results/stub_baseline.json:metrics.n--> tasks. The real `gemma3:27b` pass is
recorded by the `make record` step and committed as `eval_results.json` (`task_success` 0.90 on
the same 20 tasks). The stub exists to prove the machinery; the real model earns the green gate.

## The agent

The agent is a bounded ReAct loop. Each step it sees the question, the tool list and the
observation history, then emits one line: `ACTION: <tool> <arg>` to call a tool, or
`ANSWER: <text>` to stop. The loop stops at `ANSWER` or at `max_steps`.

Analogy: the agent is a clerk with two reference tools and a notepad. It looks something up,
writes the result on the pad, looks up the next thing, then does the sum. If it never writes a
clean instruction, the step is wasted.

Observed in this lab: the stub takes a mean of 2.0 steps per task
<!--lab:results/stub_baseline.json:metrics.mean_steps--> (one retrieve, then an answer), because
it never reaches the calc step.

## Tools behind one Protocol

A tool is anything with a `name`, a `description` and a `call(arg) -> str`. Two ship: `retrieve`
(a deterministic token-overlap lookup over the corpus) and `calc` (arithmetic over an AST
whitelist). One `ToolRegistry` is shared by the agent and the MCP server, so a tool has one
implementation and cannot drift between the two call paths.

Analogy: the tools are two labelled drawers. The agent and any outside caller reach into the
same drawers, so they always pull the same thing.

## task_success (the primary metric)

`task_success` is 1.0 when the agent's final answer matches the gold answer after
normalisation (numbers compare equal regardless of trailing zeros), else 0.0. The run reports
the mean across tasks.

Analogy: task_success is whether the final number on the notepad is right. Retrieving the facts
is not enough; the clerk still has to do the sum.

Observed in this lab: the stub scores 0.0
<!--lab:results/stub_baseline.json:metrics.task_success-->, because it answers with the raw
retrieved text and never runs the arithmetic. That is why the gate stays red until a real model
chains the tools.

## tool_precision

`tool_precision` is the fraction of a task's tool calls that were among its `required_tools`,
averaged over tasks that called a tool. It is `None` for a task that called no tool, so "used no
tools" stays distinct from "used the wrong tools".

Analogy: tool_precision is how many of the drawers the clerk opened were the right drawers. An
agent that opens the right drawers but still writes the wrong total has high precision and low
success.

Observed in this lab: the stub's `tool_precision` is 1.0
<!--lab:results/stub_baseline.json:metrics.tool_precision-->, because its one call per task is
the `retrieve` tool, which is always required. High tool precision with zero task success is the
honest signature of an agent that uses the right tool and still fails the task.

## invalid_action_rate

`invalid_action_rate` is the fraction of steps whose line parsed as neither a tool call nor an
answer. A bounded loop means a model that never emits a clean line stops at `max_steps` rather
than hanging.

Analogy: an invalid action is a note on the pad that is not an instruction the clerk can follow.
Too many and no work gets done.

Observed in this lab: the stub's `invalid_action_rate` is 0.0
<!--lab:results/stub_baseline.json:metrics.invalid_action_rate-->, because it always emits a
well-formed line. The `--force-invalid` sabotage drives this to 1.0 and the gate rejects it.

## The gate, and why it needs more than a floor

A `task_success` floor alone is not enough: a provider could pass by luck, by echoing one fixed
string, or by never using a tool. So the gate also checks provider authenticity (a stub is not
the recorded model), tool usage (at least one tool call), and non-triviality (the per-task
answers must not be byte-identical). Each failure names itself.

Analogy: a passing grade needs the right answers AND evidence the clerk actually worked: the
right drawers were opened, the answers differ by task, and a real clerk sat at the desk.

Observed in this lab: the stub run fails on `task_success` and on authenticity even with the
floors relaxed, because a stub can never be the recorded real model. [VERIFY] A capable
instruction-tuned model that reliably chains retrieve then calc clears a `task_success` floor of
0.6 on this task set; that claim is proven only once the real cassette is recorded.

## Prove the gate can fail

`make red` breaks the agent six ways and each must exit non-zero and name its metric: the stub
below the `task_success` floor, `--no-tools` (tool usage), `--hardcode-answer` (non-triviality),
`--force-invalid` (invalid action rate), empty tasks (exit 2), and a task set below `min_tasks`
(exit 2). A sabotage that slips through fails `make red` itself.

Analogy: a smoke alarm you never test is not a safety device. This lab holds a match under the
gate on purpose and fails the build if the alarm stays silent.

## MCP parity

The agent calls tools in process; an outside client calls the same tools over a tiny JSON-RPC
server. The parity test proves the MCP path returns results byte-identical to the in-process
registry for every tool across every task argument.

Analogy: whether the clerk opens the drawer or a visitor opens it through a service window, the
same item comes out.

## 10 interview questions

1. Define `task_success` and `tool_precision`. Construct an agent with high tool precision and
   zero task success, and say what that pattern tells you.
2. Why are the multi-hop tasks designed so the answer is unobtainable without chaining two tools?
   What would break if a task were answerable from the question alone?
3. The calc tool uses an AST whitelist, not `eval`. Name two payloads a cleared-builtins `eval`
   sandbox would still let through, and why the whitelist does not.
4. The gate checks provider authenticity on top of the numeric floors. What failure does that
   catch that a `task_success` floor cannot?
5. What is the non-triviality check, and which sabotage is it designed to catch?
6. Why is `tool_precision` `None` rather than 0.0 for a task that called no tool, and how does
   that change the mean?
7. The loop is bounded by `max_steps`. What failure mode does the bound prevent, and how does the
   invalid-action count surface a model that never emits a clean line?
8. Why is the committed cassette a stub recording rather than a real-model one, and why can
   replaying it never pass the gate?
9. The `ToolRegistry` is shared by the agent and the MCP server. What class of bug does that
   prevent, and how is it tested?
10. You want a gate that catches regressions without tripping on noise. How do you set a
    tolerance band, and how do you prove the gate can actually fail?
