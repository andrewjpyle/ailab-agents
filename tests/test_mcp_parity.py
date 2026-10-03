"""MCP parity: the JSON-RPC path returns results byte-identical to the in-process registry.

Both the agent and the MCP server dispatch through the one :class:`ToolRegistry`, so a tool
called over MCP MUST return exactly what ``registry.call(...)`` returns. These tests prove that
for every tool across every task argument, and smoke the real stdio loop.
"""

from __future__ import annotations

import io
import json

from ailab_agents.data import load_corpus, load_tasks
from ailab_agents.mcp_server import (
    PROTOCOL_VERSION,
    dispatch,
    handle_request,
    mcp_roundtrip,
    serve,
)
from ailab_agents.tools import ToolRegistry, build_registry
from tests.conftest import CORPUS, TASKS


def _registry() -> ToolRegistry:
    return build_registry(load_corpus(CORPUS))


def test_initialize_reports_server_info() -> None:
    result = mcp_roundtrip(_registry(), "initialize", {})
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "ailab-agents"


def test_tools_list_shape_matches_registry() -> None:
    registry = _registry()
    listed = mcp_roundtrip(registry, "tools/list", {})["tools"]
    assert [t["name"] for t in listed] == registry.names()
    for tool in listed:
        assert tool["inputSchema"]["required"] == ["input"]


def test_tools_call_is_byte_identical_for_every_task_arg() -> None:
    registry = _registry()
    args: list[tuple[str, str]] = []
    for task in load_tasks(TASKS):
        args.append(("retrieve", task.question))
        for fact in task.gold_facts:
            args.append(("retrieve", fact))
    for expr in ("1+2*3", "42+37", "(5-2)*4", "100/4", "-7+9"):
        args.append(("calc", expr))

    for name, arg in args:
        in_process = registry.call(name, arg)
        via_mcp = mcp_roundtrip(registry, "tools/call", {"name": name, "arguments": {"input": arg}})
        assert via_mcp == {"content": [{"type": "text", "text": in_process}]}


def test_tools_call_wraps_tool_error_without_raising() -> None:
    result = mcp_roundtrip(
        _registry(), "tools/call", {"name": "calc", "arguments": {"input": "1/0"}}
    )
    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("ERROR:")


def test_unknown_method_is_a_jsonrpc_error() -> None:
    response = handle_request(_registry(), {"id": 9, "method": "resources/list"})
    assert response["error"]["code"] == -32601


def test_bad_method_and_params_are_rejected() -> None:
    assert handle_request(_registry(), {"id": 1, "method": 5})["error"]["code"] == -32600
    bad = handle_request(_registry(), {"id": 1, "method": "tools/list", "params": 5})
    assert bad["error"]["code"] == -32602


def test_tools_call_requires_input() -> None:
    response = handle_request(
        _registry(), {"id": 1, "method": "tools/call", "params": {"name": "calc"}}
    )
    assert response["error"]["code"] == -32602


def test_dispatch_rejects_non_string_name() -> None:
    response = handle_request(
        _registry(),
        {"id": 1, "method": "tools/call", "params": {"name": 5, "arguments": {"input": "1"}}},
    )
    assert response["error"]["code"] == -32602


def test_serve_round_trips_line_delimited_json() -> None:
    registry = _registry()
    requests = "\n".join(
        [
            "",  # a blank line is skipped
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "calc", "arguments": {"input": "6/2"}},
                }
            ),
            "{not json",  # a parse error returns a -32700 response
        ]
    )
    sink = io.StringIO()
    assert serve(registry, stdin=io.StringIO(requests), stdout=sink) == 0
    lines = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert lines[0]["result"]["tools"][0]["name"] == "retrieve"
    assert lines[1]["result"]["content"][0]["text"] == "3"
    assert lines[2]["error"]["code"] == -32700


def test_dispatch_direct_matches_roundtrip() -> None:
    registry = _registry()
    assert dispatch(registry, "tools/list", {}) == mcp_roundtrip(registry, "tools/list", {})
