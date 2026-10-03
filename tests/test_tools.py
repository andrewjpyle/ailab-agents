import pytest

from ailab_agents.data import Record, load_corpus
from ailab_agents.tools import (
    CalcTool,
    RetrieveTool,
    Tool,
    ToolError,
    ToolRegistry,
    build_registry,
)
from tests.conftest import CORPUS

CORPUS_RECORDS = [
    Record(id="alpha-base", text="Alpha Base keeps a crew of 10 and 3 sensor arrays."),
    Record(id="bravo-base", text="Bravo Base keeps a crew of 20 and 5 sensor arrays."),
]


def test_retrieve_returns_best_overlap_record() -> None:
    tool = RetrieveTool(CORPUS_RECORDS)
    assert "Bravo Base" in tool.call("bravo base")
    assert "Alpha Base" in tool.call("alpha-base")
    assert isinstance(tool, Tool)


def test_retrieve_rejects_empty_query() -> None:
    with pytest.raises(ToolError, match="empty query"):
        RetrieveTool(CORPUS_RECORDS).call("   ")


def test_retrieve_raises_on_no_match() -> None:
    with pytest.raises(ToolError, match="no record matches"):
        RetrieveTool(CORPUS_RECORDS).call("zzzznothing")


def test_retrieve_rejects_empty_corpus() -> None:
    with pytest.raises(ToolError, match="empty"):
        RetrieveTool([])


def test_retrieve_over_bundled_corpus_is_deterministic() -> None:
    tool = RetrieveTool(load_corpus(CORPUS))
    first = tool.call("Quartz Meridian Station")
    assert first == tool.call("Quartz Meridian Station")
    assert "Quartz Meridian" in first


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("42+37", "79"),
        ("1+2*3", "7"),
        ("(1+2)*3", "9"),
        ("6/2", "3"),
        ("-5+8", "3"),
        ("100-1", "99"),
        ("7/2", "3.5"),
        (" 2 * 4 ", "8"),
    ],
)
def test_calc_known_values(expr: str, expected: str) -> None:
    assert CalcTool().call(expr) == expected


def test_calc_rejects_empty() -> None:
    with pytest.raises(ToolError, match="empty expression"):
        CalcTool().call("  ")


def test_calc_rejects_overlong() -> None:
    with pytest.raises(ToolError, match="longer than"):
        CalcTool().call("1+" * 200 + "1")


def test_calc_rejects_syntax_error() -> None:
    with pytest.raises(ToolError, match="cannot parse"):
        CalcTool().call("1 +")


def test_calc_rejects_division_by_zero() -> None:
    with pytest.raises(ToolError, match="division by zero"):
        CalcTool().call("1/0")


def test_registry_lists_and_calls() -> None:
    registry = build_registry(CORPUS_RECORDS)
    names = [item["name"] for item in registry.list()]
    assert names == ["retrieve", "calc"]
    assert registry.names() == ["retrieve", "calc"]
    assert registry.call("calc", "2+2") == "4"
    assert "Alpha Base" in registry.call("retrieve", "alpha")


def test_registry_rejects_duplicate_registration() -> None:
    registry = ToolRegistry()
    registry.register(CalcTool())
    with pytest.raises(ToolError, match="already registered"):
        registry.register(CalcTool())


def test_registry_rejects_unknown_tool() -> None:
    with pytest.raises(ToolError, match="unknown tool"):
        ToolRegistry().call("nope", "x")
