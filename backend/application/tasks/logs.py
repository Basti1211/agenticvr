"""Persistent logs for imported task executions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from application.tasks.models import TaskRunLog, TaskRunLogSummary

DEFAULT_TASK_RUN_LOG_DIR = Path(__file__).resolve().parents[2] / "task_logs"
SUMMARY_KEYS = {
    "run_id",
    "log_id",
    "setting_name",
    "task_name",
    "task_type",
    "dataset",
    "status",
    "created_at",
    "updated_at",
    "source_file",
    "max_iterations",
    "max_candidates",
    "hint_index",
    "hide_avs_submission_ratings_from_planner",
    "iteration_count",
    "final_candidate_id",
    "final_answer",
    "duration_ms",
    "error",
}
REQUIRED_SUMMARY_KEYS = {
    "run_id",
    "log_id",
    "task_name",
    "task_type",
    "dataset",
    "status",
    "max_iterations",
    "max_candidates",
}
HEAVY_LOG_KEYS = {"actions", "result", "full_history"}
SUMMARY_PREFIX_SIZES = (64 * 1024, 256 * 1024, 1024 * 1024)
_SUMMARY_CACHE: dict[Path, tuple[int, int, TaskRunLogSummary]] = {}


class _NeedMoreJson(Exception):
    """Raised when the summary prefix ends before a complete top-level value."""


def task_run_log_dir() -> Path:
    """Return the directory used for imported-task run logs."""

    return Path(os.environ.get("TASK_RUN_LOG_DIR", DEFAULT_TASK_RUN_LOG_DIR))


def task_run_log_path(log_id: str, log_dir: Path | None = None) -> Path:
    """Return the JSON file path for a task execution log id."""

    return (log_dir or task_run_log_dir()) / f"{_safe_log_id(log_id)}.json"


def save_task_run_log(log: dict[str, Any], log_dir: Path | None = None) -> Path:
    """Persist one full task execution log as indented JSON."""

    target_dir = log_dir or task_run_log_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    log["updated_at"] = datetime.now(timezone.utc).isoformat()
    target_path = task_run_log_path(str(log["log_id"]), target_dir)
    with target_path.open("w", encoding="utf-8") as log_file:
        json.dump(log, log_file, ensure_ascii=False, indent=2)
        log_file.write("\n")
    return target_path


def list_task_run_logs(log_dir: Path | None = None) -> list[TaskRunLogSummary]:
    """List available task execution logs newest first."""

    target_dir = log_dir or task_run_log_dir()
    if not target_dir.exists():
        return []

    summaries: list[TaskRunLogSummary] = []
    for path in target_dir.rglob("*.json"):
        try:
            summaries.append(_summary_from_path(path))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue

    return sorted(
        summaries,
        key=lambda summary: (
            summary.updated_at
            or summary.created_at
            or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )


def get_task_run_log(log_id: str, log_dir: Path | None = None) -> TaskRunLog | None:
    """Load one full task execution log by id."""

    target_dir = log_dir or task_run_log_dir()
    path = task_run_log_path(log_id, target_dir)
    if not path.exists():
        matches = list(target_dir.rglob(path.name)) if target_dir.exists() else []
        if not matches:
            return None
        path = max(matches, key=lambda candidate: candidate.stat().st_mtime)

    return TaskRunLog.model_validate(_read_json(path))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as log_file:
        return json.load(log_file)


def _summary_from_path(path: Path) -> TaskRunLogSummary:
    stat = path.stat()
    cached = _SUMMARY_CACHE.get(path)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    summary = _summary_from_log(_read_summary_fields(path))
    _SUMMARY_CACHE[path] = (stat.st_mtime_ns, stat.st_size, summary)
    return summary


def _read_summary_fields(path: Path) -> dict[str, Any]:
    for prefix_size in SUMMARY_PREFIX_SIZES:
        with path.open(encoding="utf-8") as log_file:
            prefix = log_file.read(prefix_size)
        try:
            return _summary_fields_from_prefix(prefix)
        except _NeedMoreJson:
            continue

    return _read_json(path)


def _summary_fields_from_prefix(prefix: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    position = _skip_whitespace(prefix, 0)
    if position >= len(prefix):
        raise _NeedMoreJson
    if prefix[position] != "{":
        raise json.JSONDecodeError("Expected object", prefix, position)

    position += 1
    summary: dict[str, Any] = {}

    while True:
        position = _skip_whitespace(prefix, position)
        if position >= len(prefix):
            raise _NeedMoreJson
        if prefix[position] == "}":
            return summary

        key, position = _decode_prefix_value(decoder, prefix, position)
        if not isinstance(key, str):
            raise json.JSONDecodeError("Expected object key", prefix, position)

        position = _skip_whitespace(prefix, position)
        if position >= len(prefix):
            raise _NeedMoreJson
        if prefix[position] != ":":
            raise json.JSONDecodeError("Expected ':' after key", prefix, position)
        position = _skip_whitespace(prefix, position + 1)

        if key in HEAVY_LOG_KEYS and REQUIRED_SUMMARY_KEYS.issubset(summary):
            return summary

        value, position = _decode_prefix_value(decoder, prefix, position)
        if key in SUMMARY_KEYS:
            summary[key] = value

        position = _skip_whitespace(prefix, position)
        if position >= len(prefix):
            raise _NeedMoreJson
        if prefix[position] == ",":
            position += 1
            continue
        if prefix[position] == "}":
            return summary
        raise json.JSONDecodeError("Expected ',' or '}'", prefix, position)


def _decode_prefix_value(
    decoder: json.JSONDecoder,
    prefix: str,
    position: int,
) -> tuple[Any, int]:
    try:
        return decoder.raw_decode(prefix, position)
    except json.JSONDecodeError as exc:
        if exc.msg.startswith("Unterminated string") or exc.pos >= len(prefix) - 1:
            raise _NeedMoreJson from exc
        raise


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t\r\n":
        position += 1
    return position


def _summary_from_log(log: dict[str, Any]) -> TaskRunLogSummary:
    return TaskRunLogSummary.model_validate(
        {key: value for key, value in log.items() if key in SUMMARY_KEYS}
    )


def _safe_log_id(log_id: str) -> str:
    safe = "".join(character for character in log_id if character.isalnum() or character in "-_.")
    if not safe:
        raise ValueError("log_id must contain at least one safe filename character.")
    return safe
