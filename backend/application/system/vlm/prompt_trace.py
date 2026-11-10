"""Helpers for recording VLM prompts without leaking media identifiers."""

from __future__ import annotations

from typing import Any

from application.types.vlm.prompt import Prompt


def prompt_trace(prompt: Prompt) -> dict[str, Any]:
    """Return full history metadata without changing what the VLM sees.

    The VLM formatter sends prompt text plus raw media bytes/URLs. It does not send
    ``ImageInput.identifier`` or ``VideoInput.identifier``. Full history keeps those
    identifiers here so developers can trace which candidate was inspected.
    """

    return {
        "system_prompt": prompt.system_prompt,
        "context_prompt": prompt.context_prompt,
        "task_text": prompt.task_text,
        "task_images": [
            {
                "identifier": image.identifier,
                "file": str(image.file),
                "identifier_visible_to_vlm": False,
            }
            for image in prompt.task_images
        ],
        "task_videos": [
            {
                "identifier": video.identifier,
                "file": str(video.file),
                "identifier_visible_to_vlm": False,
            }
            for video in prompt.task_videos
        ],
        "context_images": [
            {
                "identifier": image.identifier,
                "file": str(image.file),
                "identifier_visible_to_vlm": False,
            }
            for image in prompt.context_images
        ],
        "context_videos": [
            {
                "identifier": video.identifier,
                "file": str(video.file),
                "identifier_visible_to_vlm": False,
            }
            for video in prompt.context_videos
        ],
        "context": prompt.context.model_dump(mode="json") if prompt.context else None,
    }
