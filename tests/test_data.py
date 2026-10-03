from pathlib import Path

import pytest

from ailab_agents.data import (
    DatasetError,
    load_corpus,
    load_tasks,
    sha256_file,
)
from tests.conftest import CORPUS, TASKS


def test_bundled_corpus_loads() -> None:
    records = load_corpus(CORPUS)
    assert len(records) == 16
    assert len({r.id for r in records}) == 16
    assert all(r.text for r in records)


def test_bundled_tasks_load() -> None:
    tasks = load_tasks(TASKS)
    assert len(tasks) == 20
    multi = [t for t in tasks if "calc" in t.required_tools]
    single = [t for t in tasks if t.required_tools == ("retrieve",)]
    assert len(multi) >= 8
    assert len(single) >= 3
    assert all(t.answer and t.gold_facts for t in tasks)


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text("hello")
    assert sha256_file(f) == sha256_file(f)
    assert len(sha256_file(f)) == 64


def test_corpus_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text('{"id":"a","text":"x"}\n\n{"id":"b","text":"y"}\n')
    assert [r.id for r in load_corpus(p)] == ["a", "b"]


def test_task_answer_accepts_a_number(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"id":"q","question":"x","answer":79,"required_tools":["calc"],"gold_facts":["d"]}\n'
    )
    assert load_tasks(p)[0].answer == "79"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json\n", "invalid JSON"),
        ("[1,2]\n", "JSON object"),
        ('{"id":"a"}\n', "text"),
        ('{"id":"","text":"x"}\n', "id"),
        ("\n\n", "empty"),
    ],
)
def test_corpus_rejects_malformed(tmp_path: Path, content: str, message: str) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(content)
    with pytest.raises(DatasetError, match=message):
        load_corpus(p)


def test_corpus_rejects_duplicate_id(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_text('{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n')
    with pytest.raises(DatasetError, match="duplicate record id"):
        load_corpus(p)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"id":"q","question":"x","required_tools":["calc"],"gold_facts":["d"]}\n', "answer"),
        (
            '{"id":"q","question":"x","answer":true,"required_tools":["calc"],"gold_facts":["d"]}\n',
            "answer",
        ),
        ('{"id":"q","question":"x","answer":"1","gold_facts":["d"]}\n', "required_tools"),
        (
            '{"id":"q","question":"x","answer":"1","required_tools":[],"gold_facts":["d"]}\n',
            "required_tools",
        ),
        ('{"id":"q","question":"x","answer":"1","required_tools":["calc"]}\n', "gold_facts"),
        (
            '{"id":"q","question":"x","answer":"1","required_tools":[2],"gold_facts":["d"]}\n',
            "non-empty strings",
        ),
        ("\n\n", "empty"),
    ],
)
def test_tasks_reject_malformed(tmp_path: Path, content: str, message: str) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(content)
    with pytest.raises(DatasetError, match=message):
        load_tasks(p)


def test_tasks_reject_duplicate_id(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"id":"q","question":"a","answer":"1","required_tools":["calc"],"gold_facts":["d"]}\n'
        '{"id":"q","question":"b","answer":"2","required_tools":["calc"],"gold_facts":["d"]}\n'
    )
    with pytest.raises(DatasetError, match="duplicate task id"):
        load_tasks(p)
