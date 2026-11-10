"""Canonical text used to present task-target modalities to a VLM."""

from __future__ import annotations


def task_text_section(task_text: str | None) -> str | None:
    """Return the shared textual task section, or ``None`` when text is absent."""

    text = (task_text or "").strip()
    if not text:
        return None
    return (
        "### TASK TEXT:\n"
        "This is the textual description of the task:\n\n"
        f"{text}"
    )


def task_images_section(image_count: int) -> str | None:
    """Return the shared task-image annotation when task frames are attached."""

    if image_count <= 0:
        return None
    return (
        "### TASK IMAGES:\n"
        "The following images belong to the task and are temporally ordered visual "
        "samples from the task. Use them as visual evidence alongside the text."
    )


def task_videos_section(video_count: int) -> str | None:
    """Return the shared task-video annotation when target videos are attached."""

    if video_count <= 0:
        return None
    if video_count == 1:
        return (
            "### TASK VIDEO:\n"
            "The following video represents the target scene. Use this target scene "
            "as visual evidence."
        )
    return (
        "### TASK VIDEOS:\n"
        "The following videos represent the target scene. Use these target scenes "
        "as visual evidence."
    )


def flattened_user_text(
    *,
    task_text: str | None,
    task_image_count: int,
    task_video_count: int,
    context_prompt: str,
) -> str:
    """Render task annotations and agent context for text-only message formats."""

    parts = [
        part
        for part in (
            task_text_section(task_text),
            task_images_section(task_image_count),
            task_videos_section(task_video_count),
            context_prompt.strip() or None,
        )
        if part is not None
    ]
    return "\n\n".join(parts)
