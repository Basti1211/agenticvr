import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from application.system.vlm.task_context import (
    flattened_user_text,
    task_images_section,
    task_text_section,
    task_videos_section,
)
from application.types.vlm.prompt import Prompt


def ollama_messages_from_prompt(prompt: Prompt) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if prompt.system_prompt:
        messages.append({"role": "system", "content": prompt.system_prompt})

    user_message: dict[str, Any] = {
        "role": "user",
        "content": flattened_user_text(
            task_text=prompt.task_text,
            task_image_count=len(prompt.task_images),
            task_video_count=len(prompt.task_videos),
            context_prompt=prompt.context_prompt,
        ),
    }

    if prompt.task_images or prompt.context_images:
        user_message["images"] = [
            base64.b64encode(image.file.read_bytes()).decode("ascii")
            for image in (*prompt.task_images, *prompt.context_images)
        ]
    if prompt.task_videos or prompt.context_videos:
        raise ValueError("The Ollama chat API does not support video inputs.")

    messages.append(user_message)
    return messages


def openai_messages_from_prompt(prompt: Prompt) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if prompt.system_prompt:
        messages.append({"role": "system", "content": prompt.system_prompt})

    if not (
        prompt.task_text
        or prompt.task_images
        or prompt.task_videos
        or prompt.context_images
        or prompt.context_videos
    ):
        messages.append({"role": "user", "content": prompt.context_prompt})
        return messages

    content: list[dict[str, Any]] = []
    if (text_section := task_text_section(prompt.task_text)) is not None:
        content.append({"type": "text", "text": text_section})
    if (image_section := task_images_section(len(prompt.task_images))) is not None:
        content.append(
            {
                "type": "text",
                "text": image_section,
            }
        )
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image.file)},
            }
            for image in prompt.task_images
        )
    if (video_section := task_videos_section(len(prompt.task_videos))) is not None:
        content.append({"type": "text", "text": video_section})
        content.extend(
            {
                "type": "video_url",
                "video_url": {"url": _media_url(video.file)},
            }
            for video in prompt.task_videos
        )
    if prompt.context_prompt.strip():
        content.append({"type": "text", "text": prompt.context_prompt})
    if prompt.context_images:
        content.append(
            {
                "type": "text",
                "text": "Candidate/context images follow.",
            }
        )
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": _image_data_url(image.file)},
        }
        for image in prompt.context_images
    )
    if prompt.context_videos:
        content.append(
            {
                "type": "text",
                "text": "Candidate/context videos follow.",
            }
        )
    content.extend(
        {
            "type": "video_url",
            "video_url": {"url": _media_url(video.file)},
        }
        for video in prompt.context_videos
    )
    messages.append({"role": "user", "content": content})
    return messages


def _image_data_url(image_path: Path) -> str:
    return _data_url(image_path)


def _media_url(media_path: Path | str) -> str:
    media_url = _normalized_url(str(media_path))
    if _is_url(media_url):
        return media_url
    return _data_url(Path(media_path))


def _data_url(media_path: Path) -> str:
    mime_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
    encoded_media = base64.b64encode(media_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_media}"


def _is_url(value: str) -> bool:
    scheme = urlparse(value).scheme
    return scheme in {"http", "https", "data"}


def _normalized_url(value: str) -> str:
    if value.startswith("http:/") and not value.startswith("http://"):
        return f"http://{value.removeprefix('http:/').lstrip('/')}"
    if value.startswith("https:/") and not value.startswith("https://"):
        return f"https://{value.removeprefix('https:/').lstrip('/')}"
    return value
