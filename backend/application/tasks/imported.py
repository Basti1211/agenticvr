"""Utilities for loading downloaded VBS task definitions."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urljoin

from application.settings.settings import _data_layer_settings
from application.tasks.data_layer_paths import dataset_media_path
from application.tasks.models import (
    Dataset,
    ImportedTaskMedia,
    ImportedTaskOption,
    MediaInput,
    TaskCreate,
    TaskInput,
    TaskType,
    VideoSegment,
)

DEFAULT_IMPORTED_TASKS_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "vbs2025_tasks.json"
)
DEFAULT_IMPORTED_TASKS_DIR = DEFAULT_IMPORTED_TASKS_PATH.parent


def load_imported_tasks(
    tasks_path: str | Path = DEFAULT_IMPORTED_TASKS_PATH,
) -> list[ImportedTaskOption]:
    """Return supported tasks from one downloaded VBS task file."""

    imported_path = Path(tasks_path)
    if not imported_path.exists():
        return []

    with imported_path.open(encoding="utf-8") as task_file:
        raw_tasks = json.load(task_file)

    return [
        task
        for task in (_normalize_imported_task(raw_task) for raw_task in raw_tasks)
        if task is not None
    ]


def load_downloaded_tasks(
    tasks_dir: str | Path = DEFAULT_IMPORTED_TASKS_DIR,
) -> list[ImportedTaskOption]:
    """Return supported tasks from every downloaded VBS task file."""

    downloaded_dir = Path(tasks_dir)
    if not downloaded_dir.exists():
        return []

    return [
        task
        for tasks_path in sorted(
            downloaded_dir.glob("vbs[0-9][0-9][0-9][0-9]_tasks.json")
        )
        for task in load_imported_tasks(tasks_path)
    ]


def imported_task_to_payload(task: ImportedTaskOption) -> TaskCreate:
    """Convert one normalized imported task to the backend task payload."""

    media = None
    task_video = None
    if task.input_media is not None:
        if task.input_media.media_type == "video":
            task_video = VideoSegment(
                name=task.input_media.file,
                start_ms=task.input_media.start_ms,
                end_ms=task.input_media.end_ms,
            )
        elif task.input_media.url:
            media = MediaInput(
                name=task.input_media.url,
                media_type=task.input_media.media_type,
                mime_type=None,
                size_bytes=None,
            )

    return TaskCreate(
        task_type=task.task_type,
        dataset=task.dataset,
        input=TaskInput(
            description=task.input_text,
            media=media,
            task_video=task_video,
            answer_video=task.target_video,
        ),
        source_task_name=task.name,
    )


def _normalize_imported_task(raw_task: dict) -> ImportedTaskOption | None:
    name = str(raw_task.get("name") or "")
    task_type = _imported_task_type(name, str(raw_task.get("task_type") or ""))
    if task_type is None:
        return None

    task_input = raw_task.get("input") or {}
    target = raw_task.get("target")
    dataset = _imported_dataset(name)

    return ImportedTaskOption(
        name=name,
        task_type=task_type,
        dataset=dataset,
        input_text=task_input.get("text"),
        input_text_hints=_imported_text_hints(task_input),
        input_media=_imported_input_media(task_input, dataset),
        target_video=_imported_video_segment(target)
        if isinstance(target, dict)
        else None,
    )


def _imported_text_hints(task_input: dict) -> list[str]:
    hints = task_input.get("text_hints")
    if not isinstance(hints, list):
        return []
    return [hint for hint in hints if isinstance(hint, str)]


def _imported_task_type(name: str, raw_task_type: str) -> TaskType | None:
    normalized = f"{name} {raw_task_type}".lower()
    if "avs" in normalized:
        return TaskType.AVS
    if "kis-v" in normalized or "kisv" in normalized:
        return TaskType.VISUAL_KIS
    if "kis-t" in normalized or "kist" in normalized:
        return TaskType.TEXTUAL_KIS
    if "qa" in normalized:
        return TaskType.QA
    return None


def _imported_dataset(name: str) -> Dataset:
    normalized_name = name.lower()
    if "lhe" in normalized_name:
        return Dataset.LHE
    if "mvk" in normalized_name:
        return Dataset.MVK
    return Dataset.V3C


def _imported_input_media(
    task_input: dict,
    dataset: Dataset,
) -> ImportedTaskMedia | None:
    for media_type in ("image", "video"):
        media = task_input.get(media_type)
        if isinstance(media, dict) and media.get("file"):
            file_name = str(media["file"])
            return ImportedTaskMedia(
                media_type=media_type,
                file=file_name,
                url=_data_layer_media_url(dataset, media_type, file_name),
                start_ms=media.get("start_ms"),
                end_ms=media.get("end_ms"),
            )
    return None


def _imported_video_segment(target: dict) -> VideoSegment | None:
    file_name = target.get("file")
    if not file_name:
        return None
    return VideoSegment(
        name=str(file_name),
        start_ms=target.get("start_ms"),
        end_ms=target.get("end_ms"),
    )


def _data_layer_media_url(
    dataset: Dataset,
    media_type: str,
    file_name: str,
) -> str:
    settings = _data_layer_settings()
    base_url = (
        settings["image_path"] if media_type == "image" else settings["video_path"]
    )
    return urljoin(
        base_url.rstrip("/") + "/", f"{dataset_media_path(dataset)}/{file_name}"
    )
