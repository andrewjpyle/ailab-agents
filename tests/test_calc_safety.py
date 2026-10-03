"""Calc safety: the calc tool uses an AST whitelist, never ``eval``/``exec``.

These are the RED tests the design promises: an import, attribute access, a call, a subscript,
a name, a comprehension and the power operator must ALL raise, so a malicious or careless
expression can never reach the interpreter. A grep in CI also asserts the source has no python
``eval``.
"""

from __future__ import annotations

import inspect

import pytest

from ailab_agents import tools
from ailab_agents.tools import CalcTool, ToolError


@pytest.mark.parametrize(
    "payload",
    [
        '__import__("os").system("echo pwned")',
        "__import__('os')",
        "().__class__",
        "(1).__class__.__bases__",
        "os.getcwd()",
        "open('x')",
        "abs(-1)",
        "data[0]",
        "[x for x in range(3)]",
        "a + b",
        "2**8",
        "lambda: 1",
        "1 if True else 0",
        "x := 5",
    ],
)
def test_unsafe_payloads_raise(payload: str) -> None:
    with pytest.raises(ToolError):
        CalcTool().call(payload)


def test_import_payload_raises_specifically() -> None:
    with pytest.raises(ToolError):
        CalcTool().call('__import__("os")')


def test_attribute_access_raises() -> None:
    with pytest.raises(ToolError):
        CalcTool().call("(1).__class__")


def test_boolean_literal_is_not_a_number() -> None:
    with pytest.raises(ToolError, match="numeric literals"):
        CalcTool().call("True")


def test_source_has_no_python_eval_or_exec() -> None:
    source = inspect.getsource(tools)
    assert "eval(" not in source
    assert "exec(" not in source
