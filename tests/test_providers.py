import json
from pathlib import Path

import pytest

from ailab_agents.agent import build_prompt
from ailab_agents.providers import (
    CassetteMissError,
    LLMProvider,
    OllamaProvider,
    ReplayProvider,
    StubProvider,
    cassette_key,
)
from tests.conftest import CASSETTE

TOOLS = [
    {"name": "retrieve", "description": "look up a record"},
    {"name": "calc", "description": "do math"},
]


def test_stub_first_move_is_retrieve() -> None:
    stub = StubProvider()
    prompt = build_prompt("What is the crew of Alpha Base?", TOOLS, [])
    out = stub.complete(prompt)
    assert out.startswith("ACTION: retrieve")
    assert stub.calls == 1
    assert isinstance(stub, LLMProvider)


def test_stub_answers_with_raw_observation_after_retrieve() -> None:
    stub = StubProvider()
    prompt = build_prompt(
        "What is the crew of Alpha Base?",
        TOOLS,
        [("retrieve alpha", "Alpha Base keeps a crew of 10.")],
    )
    out = stub.complete(prompt)
    assert out == "ANSWER: Alpha Base keeps a crew of 10."


def test_stub_without_tools_answers_the_question() -> None:
    stub = StubProvider()
    prompt = build_prompt("How much?", [], [])
    assert stub.complete(prompt) == "ANSWER: How much?"


def test_cassette_key_is_stable() -> None:
    assert cassette_key("m", "p") == cassette_key("m", "p")
    assert cassette_key("m", "p") != cassette_key("m", "q")


def test_replay_miss_raises(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"provider": "stub", "model": "m", "entries": {}}))
    replay = ReplayProvider(path, model="m")
    with pytest.raises(CassetteMissError, match="cassette miss"):
        replay.complete("a prompt that was never recorded")


def test_replay_accepts_bare_mapping(tmp_path: Path) -> None:
    key = cassette_key("m", "p")
    path = tmp_path / "c.json"
    path.write_text(json.dumps({key: {"model": "m", "response": "ANSWER: hi"}}))
    assert ReplayProvider(path, model="m").complete("p") == "ANSWER: hi"


def test_stub_cassette_header_is_labeled_stub() -> None:
    cass = json.loads(CASSETTE.read_text())
    assert cass["provider"] == "stub"
    assert cass["model"] == "stub-agent-v1"
    assert "stub-recorded" in cass["note"]
    # It must NOT masquerade as the real model.
    assert cass["model"] != "gemma3:27b"


def test_replay_of_stub_cassette_reports_stub_name() -> None:
    replay = ReplayProvider(CASSETTE, name="ollama", model="stub-agent-v1")
    # The header's provider wins, so a stub cassette can never pass the authenticity gate.
    assert replay.name == "stub"
    assert replay.entry_count > 0


def test_ollama_provider_builds_localhost_url() -> None:
    p = OllamaProvider("llama3.2", host="localhost:11434")
    assert p.url == "http://localhost:11434/api/generate"
    assert p.model == "llama3.2"
    assert p.name == "ollama"
    assert isinstance(p, LLMProvider)
