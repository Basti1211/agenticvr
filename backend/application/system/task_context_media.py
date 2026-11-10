"""Helpers for attaching encoded task media to downstream VLM prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from application.tasks.models import EncodedTask
from application.types.vlm.prompt import ImageInput


def encoded_task_context_images(encoded_task: EncodedTask) -> list[ImageInput]:
    """Return sampled task frames as anonymous image inputs for VLM context."""

    frames = encoded_task.metadata.get("selected_video_frames", [])
    if not isinstance(frames, list):
        return []

    images: list[ImageInput] = []
    for frame in frames:
        frame_path = _frame_path(frame)
        if frame_path is None or not frame_path.is_file():
            continue

        images.append(
            ImageInput(
                identifier=f"task-context-frame-{len(images) + 1}",
                file=frame_path,
            )
        )
    return images

def _frame_path(frame: Any) -> Path | None:
    if not isinstance(frame, dict):
        return None
    file_value = frame.get("file")
    if not isinstance(file_value, str) or not file_value.strip():
        return None
    return Path(file_value)
