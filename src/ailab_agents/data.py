"""Dataset loading for the agents lab.

Two JSONL datasets are read here, both validated loudly so a bad fixture fails with a
``file:line`` message instead of silently shrinking the eval set.

* A corpus row is ``{"id": str, "text": str}`` (one factual record the retrieve tool serves).
* A task row is
  ``{"id", "question", "answer", "required_tools": [str, ...], "gold_facts": [str, ...]}``.
  ``answer`` is the normalized ground truth (a number or short string). ``required_tools`` lists
  the tool names a faithful solution must use; ``gold_facts`` names the record ids that hold the
  facts the question depends on.

Each dataset file also carries a sha256 of its raw bytes, recorded in the results so a number
can be traced back to the exact file that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Record:
    """One corpus record the retrieve tool can serve."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class Task:
    """One multi-hop task with a normalized ground-truth answer.

    ``required_tools`` is the set of tool names a faithful answer must use (e.g.
    ``("retrieve", "calc")`` for a retrieve-then-combine task). ``gold_facts`` names the
    record ids that carry the facts the question depends on.
    """

    id: str
    question: str
    answer: str
    required_tools: tuple[str, ...]
    gold_facts: tuple[str, ...]


class DatasetError(ValueError):
    """Raised when a dataset file is malformed."""


def sha256_file(path: str | Path) -> str:
    """Return the sha256 hex digest of a file's raw bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: Path) -> list[tuple[int, dict[str, object]]]:
    """Return ``(lineno, object)`` pairs, skipping blank lines and validating JSON."""
    rows: list[tuple[int, dict[str, object]]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{path}:{lineno}: expected a JSON object")
            rows.append((lineno, row))
    return rows


def _str_field(path: Path, lineno: int, row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{path}:{lineno}: need a non-empty string {key!r}")
    return value


def _str_list(path: Path, lineno: int, row: dict[str, object], key: str) -> tuple[str, ...]:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise DatasetError(f"{path}:{lineno}: {key!r} must be a non-empty list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise DatasetError(f"{path}:{lineno}: {key!r} must contain only non-empty strings")
        out.append(item)
    return tuple(out)


def _answer_field(path: Path, lineno: int, row: dict[str, object]) -> str:
    """Return the answer as a string. A number is accepted and stringified."""
    value = row.get("answer")
    if isinstance(value, bool) or value is None:
        raise DatasetError(f"{path}:{lineno}: need an 'answer' (string or number)")
    if isinstance(value, str):
        if not value:
            raise DatasetError(f"{path}:{lineno}: 'answer' must be a non-empty string")
        return value
    if isinstance(value, int | float):
        return str(value)
    raise DatasetError(f"{path}:{lineno}: 'answer' must be a string or number")


def load_corpus(path: str | Path) -> list[Record]:
    """Load the corpus JSONL. Raises :class:`DatasetError` on any malformed row."""
    path = Path(path)
    records: list[Record] = []
    seen: set[str] = set()
    for lineno, row in _rows(path):
        rec_id = _str_field(path, lineno, row, "id")
        if rec_id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate record id {rec_id!r}")
        seen.add(rec_id)
        records.append(Record(id=rec_id, text=_str_field(path, lineno, row, "text")))
    if not records:
        raise DatasetError(f"{path}: corpus is empty")
    return records


def load_tasks(path: str | Path) -> list[Task]:
    """Load the tasks JSONL. Raises :class:`DatasetError` on any bad row."""
    path = Path(path)
    tasks: list[Task] = []
    seen: set[str] = set()
    for lineno, row in _rows(path):
        t_id = _str_field(path, lineno, row, "id")
        if t_id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate task id {t_id!r}")
        seen.add(t_id)
        tasks.append(
            Task(
                id=t_id,
                question=_str_field(path, lineno, row, "question"),
                answer=_answer_field(path, lineno, row),
                required_tools=_str_list(path, lineno, row, "required_tools"),
                gold_facts=_str_list(path, lineno, row, "gold_facts"),
            )
        )
    if not tasks:
        raise DatasetError(f"{path}: task set is empty")
    return tasks
