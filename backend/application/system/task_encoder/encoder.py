"""Task encoder boundary.

This module defines the interface and a placeholder implementation so the rest of the
pipeline can depend on a stable contract.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from hashlib import sha256
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

from application.modalities.image import download_image
from application.modalities.video import (
    VideoClipSegment,
    VideoFrame,
    resize_video,
    sample_video_frames,
    split_and_resize_video,
)
from application.settings.settings import _data_layer_settings, _task_encoder_settings
from application.tasks.data_layer_paths import dataset_media_path
from application.system.prompts.template import render_prompt
from application.system.task_encoder.prompt import (
    TASK_ENCODER_IMAGE_DESCRIBER_CONTEXT_PROMPT,
    TASK_ENCODER_IMAGE_DESCRIBER_SYSTEM_PROMPT,
    TASK_ENCODER_QA_TEMPORAL_MODE_CLASSIFIER_CONTEXT_PROMPT,
    TASK_ENCODER_QA_TEMPORAL_MODE_CLASSIFIER_SYSTEM_PROMPT,
    TASK_ENCODER_TASK_VIDEO_DESCRIBER_CONTEXT_PROMPT,
    TASK_ENCODER_TASK_VIDEO_DESCRIBER_SYSTEM_PROMPT,
    TASK_ENCODER_VIDEO_SEGMENT_DESCRIBER_CONTEXT_PROMPT,
    TASK_ENCODER_VIDEO_SEGMENT_DESCRIBER_SYSTEM_PROMPT,
)
from application.tasks.models import EncodedTask, MediaInput, TaskCreate, TaskType, VideoSegment
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.anonymize import (
    anonymize_prompt_media_ids,
    anonymize_prompt_media_ids_batch,
)
from application.system.vlm.executor import run, run_parallel
from application.system.vlm.prompt_trace import prompt_trace
from application.types.vlm.prompt import (
    ImageInput,
    LlmAgent,
    Prompt,
    PromptContext,
    VideoInput,
)

DEFAULT_TASK_ENCODER_MAX_TOKENS = 5


class TaskEncoderSettings:
    def __init__(
        self,
        split_video_after_seconds: float,
        minimum_segment_length_seconds: float,
        resolution_width: int,
        resolution_height: int,
        max_tokens: int = DEFAULT_TASK_ENCODER_MAX_TOKENS,
        include_video_frames: bool = False,
        video_frame_count: int = 3,
    ) -> None:
        if split_video_after_seconds <= 0:
            raise ValueError("split_video_after_seconds must be greater than 0.")
        if minimum_segment_length_seconds < 1:
            raise ValueError("minimum_segment_length_seconds must be at least 1.")
        if resolution_width <= 0 or resolution_height <= 0:
            raise ValueError(
                "Video resolution width and height must be greater than 0."
            )
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        if video_frame_count <= 0:
            raise ValueError("video_frame_count must be greater than 0.")

        self.split_video_after_seconds = split_video_after_seconds
        self.minimum_segment_length_seconds = minimum_segment_length_seconds
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.max_tokens = max_tokens
        self.include_video_frames = include_video_frames
        self.video_frame_count = video_frame_count

    @classmethod
    def load(cls) -> "TaskEncoderSettings":
        settings = _task_encoder_settings()
        return cls(
            split_video_after_seconds=float(settings["split_video_after_seconds"]),
            minimum_segment_length_seconds=float(
                settings.get("minimum_segment_length_seconds", 1.0)
            ),
            resolution_width=int(settings["resolution_width"]),
            resolution_height=int(settings["resolution_height"]),
            max_tokens=int(settings.get("max_tokens", DEFAULT_TASK_ENCODER_MAX_TOKENS)),
            include_video_frames=bool(settings.get("include_video_frames", False)),
            video_frame_count=int(settings.get("video_frame_count", 3)),
        )


class TaskEncoder(ABC):
    """Maps a task submission to an encoded task representation."""

    @abstractmethod
    async def encode(self, task: TaskCreate) -> EncodedTask:
        """Encode a task for downstream retrieval components."""


class VLMTaskEncoder(TaskEncoder):
    """VLM-based task encoder implementation."""

    def __init__(
        self,
        model: str = "qwen",
        vlm_runner: Callable[[str, Prompt], Awaitable[str]] = run,
        parallel_vlm_runner: Callable[[str, list[Prompt]], Awaitable[list[str]]]
        | None = None,
        video_segmenter: Callable[
            [Path | str, float, float, int, int], list[VideoClipSegment]
        ] = split_and_resize_video,
        video_clipper: Callable[
            [Path | str, Path, int, int, float | None, float | None], Path
        ]
        | None = None,
        frame_sampler: Callable[[Path | str, int], list[VideoFrame]] = sample_video_frames,
        settings: TaskEncoderSettings | None = None,
    ) -> None:
        self.model = model
        self.vlm_runner = vlm_runner
        if parallel_vlm_runner is not None:
            self.parallel_vlm_runner = parallel_vlm_runner
        elif vlm_runner is run:
            self.parallel_vlm_runner = run_parallel
        else:
            self.parallel_vlm_runner = None
        self.video_segmenter = video_segmenter
        self.video_clipper = video_clipper or _resize_video_clip
        self.frame_sampler = frame_sampler
        self.settings = settings or TaskEncoderSettings.load()
        self._settings_were_injected = settings is not None

    async def encode(self, task: TaskCreate) -> EncodedTask:
        if not self._settings_were_injected:
            self.settings = TaskEncoderSettings.load()

        text_parts: list[str] = []
        media_refs: list[str] = []
        notes = ["Encoded by VLMTaskEncoder."]
        llm_calls: list[dict[str, Any]] = []
        selected_video_frames: list[dict[str, Any]] = []
        status: Literal["pending", "encoded"] = "encoded"

        if task.input.description:
            if task.task_type == TaskType.QA:
                text_parts.append(f"Task question:\n{task.input.description.strip()}")
            else:
                text_parts.append(f"Task description:\n{task.input.description.strip()}")

        if task.input.media:
            media_description = await self._describe_media(
                task,
                llm_calls,
                selected_video_frames,
            )
            if media_description is None:
                status = "pending"
                media_refs.append(_public_media_identifier(task.input.media))
                notes.append(
                    "Task media was submitted as metadata only "
                    "and could not be encoded without a local file or URL."
                )
            else:
                text_parts.append(f"Media description:\n{media_description.strip()}")
                notes.append("Task media was encoded as text.")

        if task.input.task_video:
            task_video_description = await self._describe_task_video(
                task,
                llm_calls,
                selected_video_frames,
            )
            text_parts.append(
                f"Task video description:\n{task_video_description.strip()}"
            )
            notes.append("Task video reference was encoded from the data layer.")

        text = "\n\n".join(text_parts) or None
        metadata: dict[str, Any] = {
            "llm_calls": llm_calls,
            "selected_video_frames": selected_video_frames,
        }
        if task.task_type == TaskType.QA and task.input.description:
            temporal_mode = await self._classify_qa_temporal_mode(
                task,
                llm_calls,
            )
            metadata["qa_temporal_mode"] = temporal_mode
            metadata["qa_time_specific"] = temporal_mode == "T"

        return EncodedTask(
            status=status,
            text=text,
            media_refs=media_refs,
            notes=notes,
            metadata=metadata,
        )

    async def _classify_qa_temporal_mode(
        self,
        task: TaskCreate,
        llm_calls: list[dict[str, Any]],
    ) -> str:
        question = (task.input.description or "").strip()
        prompt = Prompt(
            system_prompt=render_prompt(
                TASK_ENCODER_QA_TEMPORAL_MODE_CLASSIFIER_SYSTEM_PROMPT,
                task.task_type,
                task.dataset,
            ),
            context_prompt=render_prompt(
                TASK_ENCODER_QA_TEMPORAL_MODE_CLASSIFIER_CONTEXT_PROMPT,
                task.task_type,
                task.dataset,
                QUESTION=question,
            ),
            max_tokens=1,
            context_images=[],
            context_videos=[],
            context=PromptContext(
                llm_agent=LlmAgent.TASK_ENCODER_QA_TEMPORAL_MODE_CLASSIFIER,
                dataset=task.dataset,
                task_type=task.task_type,
            ),
        )

        llm_start = timer_start()
        response = await self.vlm_runner(self.model, prompt)
        duration_ms = elapsed_ms(llm_start)
        value = _parse_temporal_mode(response)
        llm_calls.append(
            {
                "phase": "qa_temporal_mode_classifier",
                "prompt": prompt_trace(prompt),
                "response": response,
                "parsed": value,
                "model": self.model,
                "duration_ms": duration_ms,
            }
        )
        return value

    async def _describe_media(
        self,
        task: TaskCreate,
        llm_calls: list[dict[str, Any]],
        selected_video_frames: list[dict[str, Any]],
    ) -> str | None:
        media = task.input.media
        if media is None:
            return ""

        media_path = _available_media_path(
            media.local_path or media.name,
            media.media_type,
        )
        if media_path is None:
            return None

        if media.media_type == "video":
            return await self._describe_video_segments(
                task,
                media_path,
                llm_calls,
                selected_video_frames,
            )

        media_identifier = _public_media_identifier(media, media_path)
        prompt = Prompt(
            system_prompt=render_prompt(
                TASK_ENCODER_IMAGE_DESCRIBER_SYSTEM_PROMPT,
                task.task_type,
                task.dataset,
            ),
            context_prompt=render_prompt(
                TASK_ENCODER_IMAGE_DESCRIBER_CONTEXT_PROMPT,
                task.task_type,
                task.dataset,
            ),
            max_tokens=self.settings.max_tokens,
            context_images=[ImageInput(identifier=media_identifier, file=media_path)]
            if media.media_type == "image"
            else [],
            context_videos=[VideoInput(identifier=media_identifier, file=media_path)]
            if media.media_type == "video"
            else [],
            context=PromptContext(
                llm_agent=LlmAgent.TASK_ENCODER_IMAGE_DESCRIBER,
                dataset=task.dataset,
                task_type=task.task_type,
            ),
        )
        vlm_prompt = anonymize_prompt_media_ids(prompt)

        llm_start = timer_start()
        response = await self.vlm_runner(self.model, vlm_prompt)
        duration_ms = elapsed_ms(llm_start)
        llm_calls.append(
            {
                "media_id": media_identifier,
                "prompt": prompt_trace(vlm_prompt),
                "response": response,
                "model": self.model,
                "duration_ms": duration_ms,
            }
        )
        return response

    async def _describe_task_video(
        self,
        task: TaskCreate,
        llm_calls: list[dict[str, Any]],
        selected_video_frames: list[dict[str, Any]],
    ) -> str:
        segment = task.input.task_video
        if segment is None:
            return ""

        source = _task_video_source(task, segment)
        prompt_video_path = _materialize_task_video_segment(
            source=source,
            dataset=task.dataset.value,
            segment=segment,
            resolution_width=self.settings.resolution_width,
            resolution_height=self.settings.resolution_height,
            video_clipper=self.video_clipper,
        )
        media_identifier = _task_video_identifier(task, segment)
        frame_inputs = self._sample_video_frame_inputs(
            source=prompt_video_path,
            source_label="task_video",
            timestamp_offset_ms=segment.start_ms or 0,
            selected_video_frames=selected_video_frames,
        )
        prompt = Prompt(
            system_prompt=render_prompt(
                TASK_ENCODER_TASK_VIDEO_DESCRIBER_SYSTEM_PROMPT,
                task.task_type,
                task.dataset,
            ),
            context_prompt=render_prompt(
                TASK_ENCODER_TASK_VIDEO_DESCRIBER_CONTEXT_PROMPT,
                task.task_type,
                task.dataset,
            ),
            max_tokens=self.settings.max_tokens,
            context_images=frame_inputs,
            context_videos=[
                VideoInput(
                    identifier=media_identifier,
                    file=prompt_video_path,
                )
            ],
            context=PromptContext(
                llm_agent=LlmAgent.TASK_ENCODER_TASK_VIDEO_DESCRIBER,
                dataset=task.dataset,
                task_type=task.task_type,
            ),
        )
        vlm_prompt = anonymize_prompt_media_ids(prompt)

        llm_start = timer_start()
        response = await self.vlm_runner(self.model, vlm_prompt)
        duration_ms = elapsed_ms(llm_start)
        llm_calls.append(
            {
                "media_id": media_identifier,
                "prompt": prompt_trace(vlm_prompt),
                "response": response,
                "model": self.model,
                "duration_ms": duration_ms,
                "source": "task_video",
            }
        )
        return response

    async def _describe_video_segments(
        self,
        task: TaskCreate,
        media_path: Path | str,
        llm_calls: list[dict[str, Any]],
        selected_video_frames: list[dict[str, Any]],
    ) -> str:
        media = task.input.media
        if media is None:
            return ""

        segments = self.video_segmenter(
            media_path,
            self.settings.split_video_after_seconds,
            self.settings.minimum_segment_length_seconds,
            self.settings.resolution_width,
            self.settings.resolution_height,
        )
        if not segments:
            raise ValueError(f"Video '{media_path}' did not produce any segments.")

        media_identifier = _public_media_identifier(media, media_path)
        prompts: list[Prompt] = []
        for segment in segments:
            frame_inputs = self._sample_video_frame_inputs(
                source=segment.path,
                source_label=(
                    f"media {media_identifier} "
                    f"{_format_seconds(segment.start_seconds)}"
                    f"-{_format_seconds(segment.end_seconds)}s"
                ),
                timestamp_offset_ms=round(segment.start_seconds * 1000),
                selected_video_frames=selected_video_frames,
            )
            prompts.append(
                Prompt(
                    system_prompt=render_prompt(
                        TASK_ENCODER_VIDEO_SEGMENT_DESCRIBER_SYSTEM_PROMPT,
                        task.task_type,
                        task.dataset,
                    ),
                    context_prompt=render_prompt(
                        TASK_ENCODER_VIDEO_SEGMENT_DESCRIBER_CONTEXT_PROMPT,
                        task.task_type,
                        task.dataset,
                    ),
                    max_tokens=self.settings.max_tokens,
                    context_images=frame_inputs,
                    context_videos=[
                        VideoInput(
                            identifier=(
                                f"{media_identifier} "
                                f"{_format_seconds(segment.start_seconds)}"
                                f"-{_format_seconds(segment.end_seconds)}s"
                            ),
                            file=segment.path,
                        ),
                    ],
                    context=PromptContext(
                        llm_agent=LlmAgent.TASK_ENCODER_VIDEO_SEGMENT_DESCRIBER,
                        dataset=task.dataset,
                        task_type=task.task_type,
                    ),
                )
            )
        vlm_prompts = anonymize_prompt_media_ids_batch(prompts)
        llm_start = timer_start()
        descriptions = await self._run_parallel(vlm_prompts)
        batch_duration_ms = elapsed_ms(llm_start)
        llm_calls.extend(
            {
                "media_id": prompt.context_videos[0].identifier,
                "prompt": prompt_trace(vlm_prompt),
                "response": description,
                "model": self.model,
                "batch_duration_ms": batch_duration_ms,
                "parallel_batch_size": len(prompts),
            }
            for prompt, vlm_prompt, description in zip(
                prompts,
                vlm_prompts,
                descriptions,
                strict=True,
            )
        )
        return "\n".join(
            (
                f"{_format_seconds(segment.start_seconds)}"
                f"-{_format_seconds(segment.end_seconds)}s: "
                f"{description.strip()}"
            )
            for segment, description in zip(segments, descriptions, strict=True)
        )

    async def _run_parallel(self, prompts: list[Prompt]) -> list[str]:
        if self.parallel_vlm_runner is None:
            return await asyncio.gather(
                *(self.vlm_runner(self.model, prompt) for prompt in prompts)
            )

        return await self.parallel_vlm_runner(self.model, prompts)

    def _sample_video_frame_inputs(
        self,
        *,
        source: Path | str,
        source_label: str,
        timestamp_offset_ms: int,
        selected_video_frames: list[dict[str, Any]],
    ) -> list[ImageInput]:
        if not self.settings.include_video_frames:
            return []

        frames = self.frame_sampler(source, self.settings.video_frame_count)
        frame_inputs: list[ImageInput] = []
        for frame in frames:
            frame_id = _video_frame_identifier(
                source_label,
                timestamp_offset_ms,
                frame.timestamp_seconds,
            )
            selected_video_frames.append(
                {
                    "id": frame_id,
                    "source": source_label,
                    "timestamp_ms": timestamp_offset_ms
                    + round(frame.timestamp_seconds * 1000),
                    "file": str(frame.path),
                }
            )
            frame_inputs.append(ImageInput(identifier=frame_id, file=frame.path))
        return frame_inputs


def _available_media_path(media_name: str, media_type: str) -> Path | str | None:
    if _is_url(media_name):
        if media_type == "image":
            return download_image(media_name)
        return media_name

    media_path = Path(media_name).expanduser()
    if media_path.exists():
        return media_path

    if not _is_path_like_media_name(media_name, media_path):
        return None

    raise ValueError(
        f"Task media '{media_name}' is not available to the backend as a local file or URL."
    )


def _parse_temporal_mode(response: str) -> str:
    value = response.strip().upper()
    if value in {"N", "T", "V"}:
        return value
    raise ValueError(
        "QA temporal-mode classifier must return exactly 'N', 'T', or 'V'. "
        f"Received: {response!r}"
    )


def _public_media_identifier(
    media: MediaInput,
    media_path: Path | str | None = None,
) -> str:
    media_type = media.media_type
    suffix = ""
    if media_path is not None and not _is_url(str(media_path)):
        suffix = Path(media_path).suffix.lower()
    if not suffix:
        suffix = Path(media.name).suffix.lower()
    if not suffix:
        suffix = ".mp4" if media_type == "video" else ".jpg"
    return f"task{suffix}"


def _task_video_source(task: TaskCreate, segment: VideoSegment) -> Path | str:
    if _is_url(segment.name):
        return segment.name

    media_path = Path(segment.name).expanduser()
    if media_path.exists():
        return media_path

    video_name = segment.name.strip().strip("/")
    if not Path(video_name).suffix:
        video_name = f"{video_name}.mp4"
    return urljoin(
        _data_layer_settings()["video_path"].rstrip("/") + "/",
        f"{dataset_media_path(task.dataset)}/{video_name}",
    )


def _materialize_task_video_segment(
    *,
    source: Path | str,
    dataset: str,
    segment: VideoSegment,
    resolution_width: int,
    resolution_height: int,
    video_clipper: Callable[
        [Path | str, Path, int, int, float | None, float | None], Path
    ],
) -> Path:
    start_seconds = _milliseconds_to_seconds(segment.start_ms)
    duration_seconds = _segment_duration_seconds(segment)
    output_path = _task_video_clip_path(
        source=source,
        dataset=dataset,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
    )
    return video_clipper(
        source,
        output_path,
        resolution_width,
        resolution_height,
        start_seconds,
        duration_seconds,
    )


def _resize_video_clip(
    source: Path | str,
    output_path: Path,
    resolution_width: int,
    resolution_height: int,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> Path:
    return resize_video(
        source,
        output_path,
        resolution_width,
        resolution_height,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )


def _task_video_clip_path(
    *,
    source: Path | str,
    dataset: str,
    start_ms: int | None,
    end_ms: int | None,
) -> Path:
    digest = sha256(
        f"{dataset}:{source}:{start_ms}:{end_ms}".encode("utf-8")
    ).hexdigest()[:16]
    return Path(gettempdir()) / "vlm-prak" / "task-videos" / f"task-video-{digest}.mp4"


def _task_video_identifier(task: TaskCreate, segment: VideoSegment) -> str:
    digest = sha256(
        f"{task.dataset.value}:{segment.name}:{segment.start_ms}:{segment.end_ms}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"task-video-{digest}"


def _video_frame_identifier(
    source_label: str,
    timestamp_offset_ms: int,
    timestamp_seconds: float,
) -> str:
    timestamp_ms = timestamp_offset_ms + round(timestamp_seconds * 1000)
    digest = sha256(f"{source_label}:{timestamp_ms}".encode("utf-8")).hexdigest()[:12]
    return f"task-frame-{digest}"


def _milliseconds_to_seconds(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1000


def _segment_duration_seconds(segment: VideoSegment) -> float | None:
    if segment.end_ms is None:
        return None
    start_ms = segment.start_ms or 0
    return (segment.end_ms - start_ms) / 1000


def _is_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _is_path_like_media_name(media_name: str, media_path: Path) -> bool:
    return (
        media_path.is_absolute()
        or media_name.startswith("~")
        or "/" in media_name
        or "\\" in media_name
    )


def _format_seconds(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")
