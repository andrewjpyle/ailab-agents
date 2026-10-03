"""A minimal stdlib JSON-RPC 2.0 server over stdio that exposes the SAME tool registry.

The point of this module is PARITY: a tool called over MCP returns a result byte-identical to
the in-process :meth:`ailab_agents.tools.ToolRegistry.call`, because both paths dispatch through
the one registry. Scope is deliberately tiny: ``initialize``, ``tools/list`` and ``tools/call``.
No resources, prompts, notifications or network.

Run it for a real stdio round-trip::

    python -m ailab_agents.mcp_server < requests.jsonl

Each request is one line of JSON; each response is one line of JSON written to stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ailab_agents.config import load_config
from ailab_agents.data import load_corpus
from ailab_agents.tools import ToolError, ToolRegistry, build_registry

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ailab-agents"
SERVER_VERSION = "0.1.0"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}


class MCPError(Exception):
    """A JSON-RPC error with a code and a message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _tools_list(registry: ToolRegistry) -> dict[str, Any]:
    return {
        "tools": [
            {"name": t["name"], "description": t["description"], "inputSchema": _INPUT_SCHEMA}
            for t in registry.list()
        ]
    }


def _tools_call(registry: ToolRegistry, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise MCPError(-32602, "tools/call needs a string 'name' and an 'arguments' object")
    if "input" not in arguments:
        raise MCPError(-32602, "tools/call arguments must include 'input'")
    try:
        text = registry.call(name, str(arguments["input"]))
    except ToolError as exc:
        return {"content": [{"type": "text", "text": f"ERROR: {exc}"}], "isError": True}
    return {"content": [{"type": "text", "text": text}]}


def dispatch(registry: ToolRegistry, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run one MCP method against the registry and return its ``result`` payload."""
    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
    if method == "tools/list":
        return _tools_list(registry)
    if method == "tools/call":
        return _tools_call(registry, params)
    raise MCPError(-32601, f"method not found: {method}")


def mcp_roundtrip(registry: ToolRegistry, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """In-process helper for the parity test: run a method and return the ``result`` payload."""
    return dispatch(registry, method, params)


def handle_request(registry: ToolRegistry, request: dict[str, Any]) -> dict[str, Any]:
    """Turn one JSON-RPC request object into a JSON-RPC response object."""
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    if not isinstance(method, str):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32600, "message": "invalid request: 'method' must be a string"},
        }
    if not isinstance(params, dict):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": "invalid request: 'params' must be an object"},
        }
    try:
        result = dispatch(registry, method, params)
    except MCPError as exc:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": exc.code, "message": exc.message},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _default_registry() -> ToolRegistry:
    config = load_config()
    return build_registry(load_corpus(config.corpus))


def serve(registry: ToolRegistry | None = None, stdin: Any = None, stdout: Any = None) -> int:
    """Read line-delimited JSON-RPC requests from stdin, write one response per line."""
    registry = registry or _default_registry()
    source = stdin or sys.stdin
    sink = stdout or sys.stdout
    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }
        else:
            response = handle_request(registry, request)
        sink.write(json.dumps(response) + "\n")
        sink.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
