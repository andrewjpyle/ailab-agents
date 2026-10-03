"""Demo: show the agent's stages on a couple of tasks, then run the eval.

It PRINTS every step the agent takes: the action it chose, the tool it called, and the
observation it got back, ending with the final answer and whether it matched the gold. The
stub brain is task-naive on purpose, so the demo also makes the gap visible: it retrieves, then
answers with raw text instead of doing the arithmetic, which is exactly why the gate needs the
real recorded model to go green.
"""

from __future__ import annotations

from collections.abc import Sequence

from ailab_agents.agent import AgentConfig, run_agent
from ailab_agents.config import load_config
from ailab_agents.data import Task, load_corpus, load_tasks
from ailab_agents.eval import main as eval_main
from ailab_agents.metrics import normalize_answer, task_success
from ailab_agents.providers import StubProvider
from ailab_agents.tools import build_registry


def _pick(tasks: Sequence[Task], *, multi_hop: bool) -> Task | None:
    want = "calc"
    for task in tasks:
        if (want in task.required_tools) is multi_hop:
            return task
    return None


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config()
    records = load_corpus(config.corpus)
    tasks = load_tasks(config.tasks)
    registry = build_registry(records)
    provider = StubProvider()
    agent_config = AgentConfig(max_steps=config.max_steps)

    print("== ailab-agents demo ==\n")
    print(
        f"corpus: {len(records)} records, {len(tasks)} tasks, "
        f"tools={[t['name'] for t in registry.list()]}, max_steps={config.max_steps}\n"
    )

    picks = (_pick(tasks, multi_hop=True), _pick(tasks, multi_hop=False))
    for task in (p for p in picks if p):
        print(f"Task [{task.id}]: {task.question}")
        print(f"    required_tools={list(task.required_tools)} gold={task.answer}")
        trajectory = run_agent(task.id, task.question, provider, registry, agent_config)
        for i, step in enumerate(trajectory.steps, start=1):
            if step.action_kind == "tool":
                print(f"    step {i}: ACTION {step.tool_name} {step.tool_arg}")
                preview = " ".join((step.observation or "").split()[:16])
                print(f"            -> {preview} ...")
            elif step.action_kind == "answer":
                print(f"    step {i}: ANSWER {step.tool_arg}")
            else:
                print(f"    step {i}: INVALID ({step.raw_output!r})")
        match = task_success(trajectory, task.answer) == 1.0
        print(
            f"    final: {trajectory.final_answer!r} "
            f"(normalized {normalize_answer(trajectory.final_answer or '')!r}) "
            f"-> {'MATCH' if match else 'MISS'} (stopped={trajectory.stopped_reason})\n"
        )

    print("== eval ==\n")
    return eval_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
