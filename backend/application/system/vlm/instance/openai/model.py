"""OpenAI Responses API adapter for the planner."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import find_dotenv, load_dotenv

from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.candidate_video_aliases import (
    anonymize_candidate_video_ids,
    deanonymize_candidate_video_ids,
)
from application.system.vlm.task_context import (
    task_images_section,
    task_text_section,
)
from application.types.vlm.prompt import Prompt


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_TIMEOUT = 120.0
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
ImageDetail = Literal["low", "high", "auto", "original"]
logger = logging.getLogger(__name__)


async def run_openai(
    model: str,
    prompt: Prompt,
    *,
    reasoning_effort: ReasoningEffort = "low",
    max_output_tokens: int = 512,
    image_detail: ImageDetail = "low",
) -> str:
    """Run one planner prompt through the OpenAI Responses API."""

    api_key = _openai_api_key()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for the GPT-backed planner. "
            "Set it in the repository's .env file."
        )

    model_prompt = anonymize_candidate_video_ids(prompt)
    payload = _openai_request(
        model,
        model_prompt,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        image_detail=image_detail,
    )
    url = _responses_url(os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL))
    timeout = float(os.getenv("OPENAI_TIMEOUT", str(DEFAULT_OPENAI_TIMEOUT)))
    start = timer_start()
    logger.info(
        "timing.openai.start model=%s agent=%s max_output_tokens=%s "
        "reasoning_effort=%s image_count=%s",
        model,
        _agent_name(prompt),
        max_output_tokens,
        reasoning_effort,
        len(prompt.task_images) + len(prompt.context_images),
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await _post_openai_request(client, url, api_key, payload)

    response_payload = response.json()
    output = deanonymize_candidate_video_ids(_assistant_text(response_payload))
    usage = response_payload.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    logger.info(
        "timing.openai.done model=%s duration_ms=%.3f input_tokens=%s "
        "output_tokens=%s total_tokens=%s response_chars=%s",
        model,
        elapsed_ms(start),
        usage.get("input_tokens", ""),
        usage.get("output_tokens", ""),
        usage.get("total_tokens", ""),
        len(output),
    )
    return output


def _openai_request(
    model: str,
    prompt: Prompt,
    *,
    reasoning_effort: ReasoningEffort,
    max_output_tokens: int,
    image_detail: ImageDetail,
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("OpenAI model must not be empty.")
    if max_output_tokens < 1:
        raise ValueError("OpenAI max_output_tokens must be at least 1.")
    if prompt.task_videos or prompt.context_videos:
        raise ValueError("The GPT-backed planner does not support video inputs.")

    content: list[dict[str, Any]] = []
    if (text_section := task_text_section(prompt.task_text)) is not None:
        content.append({"type": "input_text", "text": text_section})
    if (image_section := task_images_section(len(prompt.task_images))) is not None:
        content.append({"type": "input_text", "text": image_section})
    for image in prompt.task_images:
        content.append(
            {
                "type": "input_image",
                "image_url": _image_data_url(image.file),
                "detail": image_detail,
            }
        )
    if prompt.context_prompt.strip():
        content.append({"type": "input_text", "text": prompt.context_prompt})
    for image in prompt.context_images:
        content.append(
            {
                "type": "input_image",
                "image_url": _image_data_url(image.file),
                "detail": image_detail,
            }
        )

    return {
        "model": model,
        "instructions": prompt.system_prompt,
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
    }


async def _post_openai_request(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> httpx.Response:
    try:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise RuntimeError(
            "OpenAI request failed with status "
            f"{error.response.status_code}: {error.response.text}"
        ) from error
    except httpx.HTTPError as error:
        raise RuntimeError(f"OpenAI request failed: {error}") from error
    return response


def _assistant_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    output = response.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            texts.extend(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            )

    text = "\n".join(texts).strip()
    if text:
        return text

    error = response.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        raise RuntimeError(f"OpenAI response error: {error['message']}")

    incomplete_details = response.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        reason = incomplete_details.get("reason", "unknown")
        raise RuntimeError(f"OpenAI response was incomplete: {reason}")

    raise RuntimeError("OpenAI response did not contain assistant text.")


def _responses_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/responses"):
        return url
    return f"{url}/responses"


def _openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file, override=False)
    return os.getenv("OPENAI_API_KEY", "").strip()


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _agent_name(prompt: Prompt) -> str:
    if prompt.context is None:
        return "unknown"
    return prompt.context.llm_agent.value
