"""Replay the committed STUB cassette through the real eval machinery, with no network.

The committed cassette ``fixtures/cassettes/agent.json`` is a STUB recording for offline tests,
not a real-model recording. These tests prove the replay seam works end to end (the recorded
prompts reproduce exactly, so there are no misses), that a miss still raises, and that replaying
the stub cassette CANNOT pass the gate (its header provider is ``stub``, not the real model).
The real gemma3:27b cassette is recorded separately by the operator-gated `make record` step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ailab_agents.config import load_config
from ailab_agents.eval import evaluate
from ailab_agents.providers import CassetteMissError, ReplayProvider
from tests.conftest import CASSETTE, CORPUS, TASKS


def test_stub_cassette_is_labeled_and_complete() -> None:
    cass = json.loads(CASSETTE.read_text())
    assert cass["provider"] == "stub"
    assert cass["model"] == "stub-agent-v1"
    # Two steps per task (retrieve then answer) over 20 tasks.
    assert len(cass["entries"]) == 40


def test_replay_reproduces_every_recorded_prompt() -> None:
    cfg = load_config().__class__(corpus=CORPUS, tasks=TASKS)
    provider = ReplayProvider(CASSETTE, name="ollama", model="stub-agent-v1")
    # No CassetteMissError means every prompt the loop builds was recorded.
    result = evaluate(cfg, provider)
    assert provider.call_count > 0
    assert result["provider"] == "stub"
    # A stub cassette is not authentic, so it still fails the gate.
    assert result["authenticity"]["authentic"] is False


def test_replay_miss_raises_on_wrong_model() -> None:
    provider = ReplayProvider(CASSETTE, name="ollama", model="gemma3:27b")
    cfg = load_config().__class__(corpus=CORPUS, tasks=TASKS)
    with pytest.raises(CassetteMissError):
        evaluate(cfg, provider)


def test_cassette_has_no_host_ip_or_local_path(tmp_path: Path) -> None:
    blob = CASSETTE.read_text()
    # Build the forbidden needles from fragments so the literal tokens never appear in this
    # tracked source file (otherwise the repo's own denylist scan would flag this test).
    forbidden = [
        "OLLAMA" + "_HOST",
        "/Users" + "/",
        "." + "ts" + ".net",
        "http://",
    ]
    for needle in forbidden:
        assert needle not in blob, "cassette leaked a forbidden token"
    # A tailnet CGNAT address in the 100.64/10 range must not appear either. The pattern is
    # assembled from fragments so no sample address literal lives in this tracked file.
    cgnat = r"\b" + "100" + r"\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"
    assert not re.search(cgnat, blob)
