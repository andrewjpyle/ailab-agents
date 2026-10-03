"""ailab_agents: a small, eval-gated, local-only bounded multi-hop tool-using agent lab."""

from ailab_agents.agent import AgentConfig, Step, Trajectory, build_prompt, parse_action, run_agent
from ailab_agents.data import Record, Task, load_corpus, load_tasks
from ailab_agents.metrics import (
    invalid_action_rate,
    normalize_answer,
    task_success,
    tool_precision,
    trajectory_steps,
)
from ailab_agents.providers import (
    LLMProvider,
    OllamaProvider,
    ReplayProvider,
    StubProvider,
)
from ailab_agents.tools import (
    CalcTool,
    RetrieveTool,
    Tool,
    ToolError,
    ToolRegistry,
    build_registry,
)

__all__ = [
    "AgentConfig",
    "CalcTool",
    "LLMProvider",
    "OllamaProvider",
    "Record",
    "ReplayProvider",
    "RetrieveTool",
    "Step",
    "StubProvider",
    "Task",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "Trajectory",
    "build_prompt",
    "build_registry",
    "invalid_action_rate",
    "load_corpus",
    "load_tasks",
    "normalize_answer",
    "parse_action",
    "run_agent",
    "task_success",
    "tool_precision",
    "trajectory_steps",
]

__version__ = "0.1.0"
