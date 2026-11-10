import asyncio
import json
import logging
import os
from contextvars import ContextVar
from itertools import count
from typing import Any

import httpx

from application.settings.settings import (
    _browser_settings,
    _model_settings,
    _qa_settings,
    _vlm_api_settings,
    _vlm_endpoint,
)
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.instance.qwen.format_inputs import (
    ollama_messages_from_prompt,
    openai_messages_from_prompt,
)
from application.system.vlm.instance.qwen.format_outputs import (
    ollama_assistant_content,
    openai_assistant_content,
)
from application.types.vlm.prompt import BatchedPrompt, LlmAgent, Prompt


DEFAULT_MAX_TOKENS = 4096
VLLM_API_TYPES = {"VLLM", "VLLM_CLUSTER"}
logger = logging.getLogger(__name__)
_REQUEST_IDS = count(1)
_ACTIVE_REQUEST_ID: ContextVar[str] = ContextVar(
    "qwen_active_request_id",
    default="qwen-unknown",
)


async def run_qwen(prompt: Prompt) -> str:
    total_start = timer_start()
    api_type = _vlm_endpoint()
    vlm_api_settings = _vlm_api_settings(api_type)
    model_settings = _model_settings(vlm_api_settings["model"])
    request_id = _next_request_id()
    logger.info(
        "timing.qwen.single.start request_id=%s %s",
        request_id,
        _prompt_log_fields(prompt),
    )

    timeout = float(os.getenv("QWEN_TIMEOUT", "120"))
    async with httpx.AsyncClient(timeout=timeout) as client:
        result = await _run_qwen_with_client(
            client,
            api_type,
            model_settings,
            vlm_api_settings,
            prompt,
            request_id=request_id,
        )
    logger.info(
        "timing.qwen.single.done request_id=%s total_ms=%.3f",
        request_id,
        elapsed_ms(total_start),
    )
    return result


async def run_qwen_parallel(prompts: BatchedPrompt) -> list[str]:
    batch_start = timer_start()
    api_type = _vlm_endpoint()
    vlm_api_settings = _vlm_api_settings(api_type)
    model_settings = _model_settings(vlm_api_settings["model"])
    parallel_requests = _parallel_requests(vlm_api_settings)
    semaphore = asyncio.Semaphore(parallel_requests)
    batch_id = f"qwen-batch-{next(_REQUEST_IDS)}"
    logger.info(
        "timing.qwen.batch.start batch_id=%s prompt_count=%s "
        "parallel_requests=%s api_type=%s",
        batch_id,
        len(prompts.prompts),
        parallel_requests,
        api_type,
    )

    timeout = float(os.getenv("QWEN_TIMEOUT", "120"))
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _run_qwen_with_limit(
                semaphore,
                client,
                api_type,
                model_settings,
                vlm_api_settings,
                prompt,
                request_id=f"{batch_id}.{index}",
            )
            for index, prompt in enumerate(prompts.prompts, start=1)
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            logger.exception(
                "timing.qwen.batch.failed batch_id=%s prompt_count=%s total_ms=%.3f",
                batch_id,
                len(prompts.prompts),
                elapsed_ms(batch_start),
            )
            raise
    logger.info(
        "timing.qwen.batch.done batch_id=%s prompt_count=%s total_ms=%.3f",
        batch_id,
        len(prompts.prompts),
        elapsed_ms(batch_start),
    )
    return results


async def _run_qwen_with_limit(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    api_type: str,
    model_settings: dict[str, Any],
    vlm_api_settings: dict[str, Any],
    prompt: Prompt,
    *,
    request_id: str,
) -> str:
    queue_start = timer_start()
    logger.info(
        "timing.qwen.queue.start request_id=%s %s",
        request_id,
        _prompt_log_fields(prompt),
    )
    await semaphore.acquire()
    queue_ms = elapsed_ms(queue_start)
    logger.info(
        "timing.qwen.queue.done request_id=%s wait_ms=%.3f",
        request_id,
        queue_ms,
    )
    try:
        return await _run_qwen_with_client(
            client,
            api_type,
            model_settings,
            vlm_api_settings,
            prompt,
            request_id=request_id,
        )
    finally:
        semaphore.release()


async def _run_qwen_with_client(
    client: httpx.AsyncClient,
    api_type: str,
    model_settings: dict[str, Any],
    vlm_api_settings: dict[str, Any],
    prompt: Prompt,
    *,
    request_id: str,
) -> str:
    total_start = timer_start()
    build_start = timer_start()
    try:
        url, payload = _qwen_request(
            api_type,
            model_settings,
            vlm_api_settings,
            prompt,
        )
    except Exception:
        logger.exception(
            "timing.qwen.request_build.failed request_id=%s build_ms=%.3f %s",
            request_id,
            elapsed_ms(build_start),
            _prompt_log_fields(prompt),
        )
        raise
    build_ms = elapsed_ms(build_start)
    media_stats = _payload_media_stats(payload)
    logger.info(
        "timing.qwen.request_built request_id=%s build_ms=%.3f "
        "payload_media_count=%s payload_data_url_chars=%s payload_remote_urls=%s "
        "prompt_max_tokens=%s submitted_max_tokens=%s thinking=%s %s",
        request_id,
        build_ms,
        media_stats["media_count"],
        media_stats["data_url_chars"],
        media_stats["remote_url_count"],
        prompt.max_tokens,
        _submitted_max_tokens(api_type, payload),
        _thinking_enabled(model_settings, prompt),
        _prompt_log_fields(prompt),
    )

    http_start = timer_start()
    logger.info(
        "timing.qwen.http.start request_id=%s url=%s",
        request_id,
        url,
    )
    request_id_token = _ACTIVE_REQUEST_ID.set(request_id)
    try:
        try:
            response = await _post_qwen_request_async(
                client,
                url,
                vlm_api_settings,
                payload,
            )
        finally:
            _ACTIVE_REQUEST_ID.reset(request_id_token)
    except Exception:
        logger.exception(
            "timing.qwen.http.failed request_id=%s duration_ms=%.3f",
            request_id,
            elapsed_ms(http_start),
        )
        raise
    http_ms = elapsed_ms(http_start)

    decode_start = timer_start()
    try:
        response_payload = response.json()
        content = _assistant_content(api_type, response_payload)
    except Exception:
        logger.exception(
            "timing.qwen.response_decode.failed request_id=%s decode_ms=%.3f",
            request_id,
            elapsed_ms(decode_start),
        )
        raise
    decode_ms = elapsed_ms(decode_start)
    usage = _response_usage(response_payload)
    headers = getattr(response, "headers", {})
    logger.info(
        "timing.qwen.http.done request_id=%s http_ms=%.3f status_code=%s "
        "server_request_id=%s server_timing=%r server_process_time=%r "
        "prompt_tokens=%s completion_tokens=%s "
        "total_tokens=%s response_chars=%s decode_ms=%.3f total_ms=%.3f",
        request_id,
        http_ms,
        getattr(response, "status_code", "stub"),
        headers.get("x-request-id", ""),
        headers.get("server-timing", ""),
        headers.get("x-process-time", ""),
        usage.get("prompt_tokens", ""),
        usage.get("completion_tokens", ""),
        usage.get("total_tokens", ""),
        len(content),
        decode_ms,
        elapsed_ms(total_start),
    )
    return content


def _qwen_request(
    api_type: str,
    model_settings: dict[str, Any],
    vlm_api_settings: dict[str, Any],
    prompt: Prompt,
) -> tuple[str, dict[str, Any]]:
    thinking_enabled = _thinking_enabled(model_settings, prompt)

    if api_type == "OLLAMA":
        payload = {
            "model": _model_name(vlm_api_settings),
            "messages": ollama_messages_from_prompt(prompt),
            "think": thinking_enabled,
            "stream": model_settings["stream"],
            "options": {
                "num_predict": _max_tokens(model_settings, prompt),
                "temperature": _temperature(model_settings, prompt),
            },
        }
        return vlm_api_settings["API_url"], payload

    if api_type in VLLM_API_TYPES:
        payload = {
            "model": _model_name(vlm_api_settings),
            "messages": openai_messages_from_prompt(prompt),
            "chat_template_kwargs": {"enable_thinking": thinking_enabled},
            "max_tokens": _max_tokens(model_settings, prompt),
            "temperature": _temperature(model_settings, prompt),
            "top_p": float(_generation_setting(model_settings, "top_p", 1.0)),
            "seed": int(_generation_setting(model_settings, "seed", 42)),
            "stream": bool(model_settings.get("stream", False)),
        }
        video_sampling_fps = _video_sampling_fps(prompt)
        if video_sampling_fps is not None:
            payload["mm_processor_kwargs"] = {
                "do_sample_frames": False,
            }
            payload["media_io_kwargs"] = {
                "video": {
                    "num_frames": -1,
                    "fps": video_sampling_fps,
                }
            }
        return (
            _chat_completions_url(vlm_api_settings["API_url"]),
            payload,
        )

    raise ValueError(f"Unsupported Qwen API type: {api_type}")


async def _post_qwen_request_async(
    client: httpx.AsyncClient,
    url: str,
    vlm_api_settings: dict[str, Any],
    payload: dict[str, Any],
) -> httpx.Response:
    request_id = _ACTIVE_REQUEST_ID.get()
    serialize_start = timer_start()
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    logger.info(
        "timing.qwen.serialize.done request_id=%s serialized_bytes=%s "
        "serialize_ms=%.3f",
        request_id,
        len(serialized_payload),
        elapsed_ms(serialize_start),
    )

    trace_start = timer_start()
    request_body_sent_at: float | None = None

    async def trace(event_name: str, info: dict[str, Any]) -> None:
        del info
        nonlocal request_body_sent_at
        if event_name.endswith("send_request_body.complete"):
            request_body_sent_at = timer_start()
            logger.info(
                "timing.qwen.transport.upload_done request_id=%s "
                "transport_total_ms=%.3f",
                request_id,
                elapsed_ms(trace_start),
            )
        elif event_name.endswith("receive_response_headers.complete"):
            server_wait_ms = (
                elapsed_ms(request_body_sent_at)
                if request_body_sent_at is not None
                else -1.0
            )
            logger.info(
                "timing.qwen.transport.headers_done request_id=%s "
                "after_upload_ms=%.3f transport_total_ms=%.3f",
                request_id,
                server_wait_ms,
                elapsed_ms(trace_start),
            )
        elif event_name.endswith("receive_response_body.complete"):
            logger.info(
                "timing.qwen.transport.response_body_done request_id=%s "
                "transport_total_ms=%.3f",
                request_id,
                elapsed_ms(trace_start),
            )
        elif event_name.endswith(".failed"):
            logger.warning(
                "timing.qwen.transport.failed request_id=%s event=%s "
                "transport_total_ms=%.3f",
                request_id,
                event_name,
                elapsed_ms(trace_start),
            )

    try:
        response = await client.post(
            url.strip(),
            headers=_headers(vlm_api_settings),
            content=serialized_payload,
            extensions={"trace": trace},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Qwen request failed with status {exc.response.status_code}: "
            f"{exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Qwen request failed: {exc}") from exc

    return response


def _assistant_content(api_type: str, response: dict[str, Any]) -> str:
    if api_type == "OLLAMA":
        return ollama_assistant_content(response)
    if api_type in VLLM_API_TYPES:
        return openai_assistant_content(response)
    raise ValueError(f"Unsupported Qwen API type: {api_type}")


def _headers(vlm_api_settings: dict[str, Any]) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = vlm_api_settings.get("API_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _chat_completions_url(api_url: str) -> str:
    url = api_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _model_name(vlm_api_settings: dict[str, Any]) -> str:
    model_name = vlm_api_settings.get("served_model") or vlm_api_settings.get("model")
    if not isinstance(model_name, str):
        raise ValueError("VLM endpoint must reference a model.")
    return model_name


def _max_tokens(model_settings: dict[str, Any], prompt: Prompt) -> int:
    """Apply the prompt-specific generation limit within the model's ceiling."""

    model_max_tokens = int(model_settings.get("max_tokens", DEFAULT_MAX_TOKENS))
    return min(model_max_tokens, prompt.max_tokens)


def _generation_setting(
    model_settings: dict[str, Any], key: str, default: float | int
) -> float | int:
    return model_settings.get(key, default)


def _temperature(model_settings: dict[str, Any], prompt: Prompt) -> float:
    if prompt.temperature is not None:
        return prompt.temperature
    return float(_generation_setting(model_settings, "temperature", 0.0))


def _video_sampling_fps(prompt: Prompt) -> float | None:
    if not (prompt.task_videos or prompt.context_videos) or prompt.context is None:
        return None

    if prompt.context.video_sampling_fps is not None:
        return prompt.context.video_sampling_fps

    agent = prompt.context.llm_agent
    if agent == LlmAgent.BROWSER_CLASSIFIER:
        fps = float(_browser_settings()["clip_frame_rate"])
        setting_name = "Browser clip"
    elif agent == LlmAgent.QA:
        fps = float(_qa_settings()["fps"])
        setting_name = "QA video"
    elif agent == LlmAgent.VQA:
        fps = float(_qa_settings()["vqa_fps"])
        setting_name = "VQA video"
    else:
        return None

    if fps <= 0:
        raise ValueError(f"{setting_name} fps must be greater than 0.")
    return fps


def _thinking_enabled(
    model_settings: dict[str, Any],
    prompt: Prompt,
) -> bool:
    think_by_agent = model_settings.get("think", {})
    if not isinstance(think_by_agent, dict):
        raise ValueError("Model setting 'think' must be an agent mapping.")
    if prompt.context is None:
        return False
    return bool(think_by_agent.get(prompt.context.llm_agent.value, False))


def _parallel_requests(vlm_api_settings: dict[str, Any]) -> int:
    parallel_requests = int(vlm_api_settings.get("parallel_requests", 1))
    if parallel_requests < 1:
        raise ValueError("VLM parallel_requests must be at least 1.")
    return parallel_requests


def _next_request_id() -> str:
    return f"qwen-{next(_REQUEST_IDS)}"


def _prompt_log_fields(prompt: Prompt) -> str:
    agent = prompt.context.llm_agent.value if prompt.context is not None else "unknown"
    task_image_ids = [image.identifier for image in prompt.task_images]
    task_video_ids = [video.identifier for video in prompt.task_videos]
    image_ids = [image.identifier for image in prompt.context_images]
    video_ids = [video.identifier for video in prompt.context_videos]
    task_image_bytes = sum(_safe_file_size(image.file) for image in prompt.task_images)
    task_video_bytes = sum(_safe_file_size(video.file) for video in prompt.task_videos)
    image_bytes = sum(_safe_file_size(image.file) for image in prompt.context_images)
    video_bytes = sum(_safe_file_size(video.file) for video in prompt.context_videos)
    return (
        f"agent={agent} task_image_ids={task_image_ids} "
        f"task_video_ids={task_video_ids} image_ids={image_ids} "
        f"video_ids={video_ids} task_image_bytes={task_image_bytes} "
        f"task_video_bytes={task_video_bytes} image_bytes={image_bytes} "
        f"video_bytes={video_bytes}"
    )


def _safe_file_size(path: Any) -> int:
    try:
        return int(path.stat().st_size)
    except (AttributeError, OSError):
        return 0


def _payload_media_stats(payload: dict[str, Any]) -> dict[str, int]:
    media_count = 0
    data_url_chars = 0
    remote_url_count = 0

    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        images = message.get("images", [])
        if isinstance(images, list):
            media_count += len(images)
            data_url_chars += sum(
                len(image) for image in images if isinstance(image, str)
            )

        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            media_type = item.get("type")
            if media_type not in {"image_url", "video_url"}:
                continue
            media_count += 1
            media_container = item.get(media_type, {})
            if not isinstance(media_container, dict):
                continue
            url = media_container.get("url")
            if not isinstance(url, str):
                continue
            if url.startswith("data:"):
                data_url_chars += len(url)
            elif url.startswith(("http://", "https://")):
                remote_url_count += 1

    return {
        "media_count": media_count,
        "data_url_chars": data_url_chars,
        "remote_url_count": remote_url_count,
    }


def _submitted_max_tokens(api_type: str, payload: dict[str, Any]) -> Any:
    if api_type == "OLLAMA":
        options = payload.get("options", {})
        return options.get("num_predict", "") if isinstance(options, dict) else ""
    return payload.get("max_tokens", "")


def _response_usage(response_payload: dict[str, Any]) -> dict[str, Any]:
    usage = response_payload.get("usage")
    if isinstance(usage, dict):
        return usage

    prompt_tokens = response_payload.get("prompt_eval_count", "")
    completion_tokens = response_payload.get("eval_count", "")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens: int | str = prompt_tokens + completion_tokens
    else:
        total_tokens = ""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
