"""Tools the agent can call, behind one narrow Protocol.

A tool is anything with a ``name``, a ``description`` and a ``call(arg) -> str``. The
:class:`ToolRegistry` is the SINGLE source of truth both the in-process agent and the MCP
server use, so there is exactly one implementation of each tool and the two paths cannot drift.

Two tools ship:

* :class:`RetrieveTool` does a deterministic token-overlap lookup over the corpus and returns
  the single best-matching record's text. No index, no network, stdlib only.
* :class:`CalcTool` evaluates a small arithmetic expression with an AST WHITELIST, never
  ``eval``/``exec``. Only numeric literals and ``+ - * /`` (and unary minus) are allowed; a name,
  a call, an attribute, a subscript, a comprehension or a dunder raises :class:`ToolError`.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from ailab_agents.data import Record
from ailab_agents.metrics import normalize_answer

_TOKEN = re.compile(r"[a-z0-9]+")
CALC_MAX_LEN = 200


class ToolError(RuntimeError):
    """Raised when a tool cannot serve a call (bad query, unsafe expression, unknown tool)."""


@runtime_checkable
class Tool(Protocol):
    """The minimum a tool must offer. Keeping the seam this narrow is what lets the agent and
    the MCP server dispatch any tool through identical code."""

    name: str
    description: str

    def call(self, arg: str) -> str: ...


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class RetrieveTool:
    """Deterministic token-overlap lookup over the corpus; returns the best record's text.

    The score is the sum of query-token frequencies found in the record, so a record that
    mentions the queried entity more often ranks higher. Ties break on the record id, so the
    result is stable across runs and platforms.
    """

    name = "retrieve"
    description = (
        "Look up a factual record by keywords. Input: a query string. Returns record text."
    )

    def __init__(self, corpus: list[Record]) -> None:
        if not corpus:
            raise ToolError("retrieve: corpus is empty")
        self._records = list(corpus)
        self._tokens = {rec.id: Counter(_tokenize(rec.text)) for rec in self._records}

    def _score(self, query_tokens: list[str], rec_id: str) -> int:
        counts = self._tokens[rec_id]
        return sum(counts.get(token, 0) for token in query_tokens)

    def call(self, arg: str) -> str:
        query_tokens = _tokenize(arg)
        if not query_tokens:
            raise ToolError("retrieve: empty query")
        # Highest overlap score wins; ties break on the record id ascending (stable).
        best = min(self._records, key=lambda rec: (-self._score(query_tokens, rec.id), rec.id))
        if self._score(query_tokens, best.id) == 0:
            raise ToolError(f"retrieve: no record matches {arg!r}")
        return best.text


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


class CalcTool:
    """Evaluate a small arithmetic expression with an AST whitelist (never ``eval``/``exec``).

    Allowed: numeric literals and the binary operators ``+ - * /`` plus unary minus. Anything
    else (a name, a call, an attribute, a subscript, a comprehension, a dunder) raises
    :class:`ToolError`. The input length is capped so a pathological expression cannot blow up
    the parser.
    """

    name = "calc"
    description = (
        "Evaluate arithmetic over + - * / and parentheses. Input: an expression. Returns a number."
    )

    def call(self, arg: str) -> str:
        expr = arg.strip()
        if not expr:
            raise ToolError("calc: empty expression")
        if len(expr) > CALC_MAX_LEN:
            raise ToolError(f"calc: expression longer than {CALC_MAX_LEN} characters")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"calc: cannot parse {expr!r} ({exc.msg})") from exc
        try:
            value = self._reduce(tree.body)
        except ZeroDivisionError as exc:
            raise ToolError("calc: division by zero") from exc
        return normalize_answer(repr(value))

    def _reduce(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ToolError(f"calc: only numeric literals are allowed, not {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ast.USub):
                raise ToolError("calc: only unary minus is allowed")
            return -self._reduce(node.operand)
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ToolError(f"calc: operator {type(node.op).__name__} is not allowed")
            left = self._reduce(node.left)
            right = self._reduce(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ToolError(f"calc: {type(node).__name__} is not allowed")


class ToolRegistry:
    """An ordered collection of tools, dispatched by name.

    This is the one registry both the agent loop and the MCP server read, so a tool behaves
    identically whether it is called in process or over JSON-RPC.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def list(self) -> list[dict[str, str]]:
        """Ordered ``[{"name", "description"}]`` for the prompt and for ``tools/list``."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def call(self, name: str, arg: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}")
        return tool.call(arg)


def build_registry(corpus: list[Record]) -> ToolRegistry:
    """Build the canonical registry: a retrieve tool over ``corpus`` plus the calc tool."""
    registry = ToolRegistry()
    registry.register(RetrieveTool(corpus))
    registry.register(CalcTool())
    return registry
