"""Configurable task-input visibility for loop modules."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from application.settings.settings import _task_input_access_settings
from application.system.task_context_media import encoded_task_context_images
from application.tasks.models import EncodedTask, MediaInput, TaskCreate, TaskType
from application.types.vlm.prompt import ImageInput, VideoInput


MODULE_NAMES = ("planner", "browser", "vqa", "answerer")
ANNOTATIONS_DIR = Path(__file__).resolve().parents[2] / "annotations"
TASK_MAPPING_FILE = ANNOTATIONS_DIR / "task_mapping.csv"
_ANNOTATOR_DIR = re.compile(r"A(?P<number>[1-9][0-9]*)")


@dataclass(frozen=True)
class ModuleTaskInputAccess:
    """Which original task-input sources a module may see."""

    description: bool
    frames: bool
    video: bool
    human1: bool = False
    human2: bool = False

    def __post_init__(self) -> None:
        enabled_text_sources = sum((self.description, self.human1, self.human2))
        if enabled_text_sources > 1:
            raise ValueError(
                "Task input text sources 'description', 'human1', and 'human2' "
                "are mutually exclusive."
            )

    @property
    def has_text_source(self) -> bool:
        return self.description or self.human1 or self.human2


@dataclass(frozen=True)
class TaskInputAccessSettings:
    planner: ModuleTaskInputAccess
    browser: ModuleTaskInputAccess
    vqa: ModuleTaskInputAccess
    answerer: ModuleTaskInputAccess = ModuleTaskInputAccess(
        description=True,
        frames=True,
        video=False,
    )


DEFAULT_TASK_INPUT_ACCESS = TaskInputAccessSettings(
    planner=ModuleTaskInputAccess(description=True, frames=True, video=False),
    browser=ModuleTaskInputAccess(description=True, frames=True, video=False),
    vqa=ModuleTaskInputAccess(description=True, frames=False, video=False),
    answerer=ModuleTaskInputAccess(description=True, frames=True, video=False),
)

_TASK_QUESTION_SECTION = re.compile(
    r"\ATask question:\s*.*?(?=^[^\n]+:\s*$|\Z)",
    flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def remove_task_question(text: str) -> str:
    """Remove a leading encoded task-question section, including at end of text."""

    return _TASK_QUESTION_SECTION.sub("", text.strip(), count=1).strip()


def load_task_input_access_settings() -> TaskInputAccessSettings:
    """Load per-module task-input source visibility."""

    try:
        settings = _task_input_access_settings()
    except FileNotFoundError:
        return DEFAULT_TASK_INPUT_ACCESS

    if not isinstance(settings, dict):
        raise ValueError("Task input access settings must be a JSON object.")

    default_access = (
        _access_from_mapping(
            settings["default"],
            fallback=None,
            label="default",
        )
        if "default" in settings
        else None
    )
    modules = settings.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Task input access settings 'modules' must be an object.")

    loaded: dict[str, ModuleTaskInputAccess] = {}
    for module_name in MODULE_NAMES:
        fallback = default_access or getattr(DEFAULT_TASK_INPUT_ACCESS, module_name)
        loaded[module_name] = _access_from_mapping(
            modules.get(module_name),
            fallback=fallback,
            label=module_name,
        )

    unknown = set(modules) - set(MODULE_NAMES)
    if unknown:
        raise ValueError(
            "Unknown task input access module(s): "
            f"{', '.join(sorted(str(item) for item in unknown))}."
        )

    return TaskInputAccessSettings(**loaded)


def encoded_task_for_module(
    encoded_task: EncodedTask,
    task: TaskCreate,
    access: ModuleTaskInputAccess,
) -> EncodedTask:
    """Return an encoded-task view containing only allowed task-input sources."""

    metadata = dict(encoded_task.metadata)
    if not access.frames:
        metadata.pop("selected_video_frames", None)
    human_slot = _human_slot(access)
    text = (
        _human_annotation_description(task, human_slot)
        if human_slot is not None
        else (
            remove_task_question(encoded_task.text) or None
            if access.description and encoded_task.text
            else None
        )
    )

    return encoded_task.model_copy(
        update={
            "text": text,
            "media_refs": (
                encoded_task.media_refs
                if access.has_text_source or access.frames or access.video
                else []
            ),
            "metadata": metadata,
        },
        deep=True,
    )


def task_description_for_module(
    encoded_task: EncodedTask,
    task: TaskCreate,
    access: ModuleTaskInputAccess,
) -> str:
    human_slot = _human_slot(access)
    if human_slot is not None:
        return _human_annotation_description(task, human_slot)
    if not access.description:
        return ""
    if encoded_task.text:
        return remove_task_question(encoded_task.text)
    if task.task_type == TaskType.QA:
        return ""
    return (task.input.description or "").strip()


def _human_slot(access: ModuleTaskInputAccess) -> int | None:
    if access.human1:
        return 1
    if access.human2:
        return 2
    return None


def task_images_for_module(
    encoded_task: EncodedTask,
    access: ModuleTaskInputAccess,
) -> list[ImageInput]:
    if not access.frames:
        return []
    return encoded_task_context_images(encoded_task)


def task_for_module(
    task: TaskCreate,
    encoded_task: EncodedTask,
    access: ModuleTaskInputAccess,
) -> TaskCreate:
    """Return a task view with disallowed original input fields removed."""

    description = task_description_for_module(encoded_task, task, access) or None
    media = _media_for_module(task.input.media, access)
    task_video = task.input.task_video if access.video else None
    task_input = task.input.model_copy(
        update={
            "description": description,
            "media": media,
            "task_video": task_video,
        },
        deep=True,
    )
    task_type = task.task_type
    if task_type == TaskType.VISUAL_KIS and media is None and task_video is None:
        # Keep the access-filtered task valid after hiding visual target media.
        task_type = TaskType.TEXTUAL_KIS
        if task_input.description is None:
            task_input = task_input.model_copy(
                update={"description": "Visual target evidence is not available."},
                deep=True,
            )
    return task.model_copy(
        update={"task_type": task_type, "input": task_input},
        deep=True,
    )


def task_videos_for_module(
    task: TaskCreate,
    access: ModuleTaskInputAccess,
    resolver: Any,
) -> list[VideoInput]:
    if not access.video:
        return []
    if task.input.task_video is None and not (
        task.input.media is not None and task.input.media.media_type == "video"
    ):
        return []
    video = resolver(task)
    return [video] if video is not None else []


def _media_for_module(
    media: MediaInput | None,
    access: ModuleTaskInputAccess,
) -> MediaInput | None:
    if media is None:
        return None
    if media.media_type == "video":
        return media if access.video else None
    return None


def _access_from_mapping(
    value: object,
    *,
    fallback: ModuleTaskInputAccess | None,
    label: str,
) -> ModuleTaskInputAccess:
    if value is None:
        if fallback is None:
            raise ValueError(f"Task input access {label!r} must be configured.")
        return fallback
    if not isinstance(value, Mapping):
        raise ValueError(f"Task input access {label!r} must be an object.")

    source: dict[str, Any] = {}
    if fallback is not None:
        source.update(
            {
                "description": fallback.description,
                "frames": fallback.frames,
                "video": fallback.video,
                "human1": fallback.human1,
                "human2": fallback.human2,
            }
        )
    source.update(value)
    source.setdefault("human1", False)
    source.setdefault("human2", False)

    fields = {"description", "frames", "video", "human1", "human2"}
    unknown = set(source) - fields
    if unknown:
        raise ValueError(
            f"Unknown task input access field(s) for {label}: "
            f"{', '.join(sorted(str(item) for item in unknown))}."
        )

    return ModuleTaskInputAccess(
        description=_bool_setting(source, "description", label),
        frames=_bool_setting(source, "frames", label),
        video=_bool_setting(source, "video", label),
        human1=_bool_setting(source, "human1", label),
        human2=_bool_setting(source, "human2", label),
    )


def _human_annotation_description(task: TaskCreate, human_slot: int) -> str:
    task_name = (task.source_task_name or "").strip()
    if not task_name:
        raise ValueError(f"Cannot use human{human_slot}: task has no source_task_name.")

    task_mapping = _load_task_mapping(TASK_MAPPING_FILE)
    try:
        video_id = task_mapping[task_name]
    except KeyError as error:
        raise ValueError(
            f"Cannot use human{human_slot}: task {task_name!r} is not present in "
            f"{TASK_MAPPING_FILE}."
        ) from error

    annotations = _annotation_files(video_id)
    if len(annotations) != 2:
        raise ValueError(
            f"Cannot use human{human_slot} for task {task_name!r}: expected exactly "
            f"two annotations for {video_id!r}, found {len(annotations)}."
        )
    annotation_file = annotations[human_slot - 1]
    try:
        with annotation_file.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot read human{human_slot} annotation {annotation_file}."
        ) from error

    annotation = payload.get("annotation") if isinstance(payload, dict) else None
    description = (
        annotation.get("description") if isinstance(annotation, dict) else None
    )
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"Human annotation {annotation_file} has no non-empty description."
        )
    return description.strip()


@lru_cache(maxsize=1)
def _load_task_mapping(mapping_file: Path) -> dict[str, str]:
    try:
        with mapping_file.open(encoding="utf-8", newline="") as file:
            rows = csv.DictReader(file)
            if rows.fieldnames is None or not {"video_id", "task_name"}.issubset(
                rows.fieldnames
            ):
                raise ValueError(
                    f"Task annotation mapping {mapping_file} must contain "
                    "'video_id' and 'task_name' columns."
                )
            mapping: dict[str, str] = {}
            for row in rows:
                task_name = (row.get("task_name") or "").strip()
                video_id = (row.get("video_id") or "").strip()
                if not task_name or not video_id:
                    raise ValueError(
                        f"Task annotation mapping {mapping_file} contains an empty "
                        "task_name or video_id."
                    )
                if task_name in mapping:
                    raise ValueError(
                        f"Task annotation mapping {mapping_file} contains duplicate "
                        f"task name {task_name!r}."
                    )
                mapping[task_name] = video_id
    except OSError as error:
        raise ValueError(
            f"Cannot read task annotation mapping {mapping_file}."
        ) from error
    return mapping


def _annotation_files(video_id: str) -> list[Path]:
    numbered_files: list[tuple[int, Path]] = []
    for annotation_file in ANNOTATIONS_DIR.glob(f"A*/{video_id}.json"):
        match = _ANNOTATOR_DIR.fullmatch(annotation_file.parent.name)
        if match is not None:
            numbered_files.append((int(match.group("number")), annotation_file))
    return [path for _, path in sorted(numbered_files)]


def _bool_setting(settings: Mapping[str, Any], name: str, label: str) -> bool:
    value = settings.get(name)
    if not isinstance(value, bool):
        raise ValueError(
            f"Task input access {label!r} field {name!r} must be a boolean."
        )
    return value
