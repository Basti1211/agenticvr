"""Independent VLM-backed verification of one keyframe-centred video scene."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from application.system.retrieval.endpoints import get_endpoint_for_query
from application.system.retrieval.fetch import fetch_candidates
from application.system.retrieval.queries.loadVideoSummary import (
    create_load_video_summary_payload,
)
from application.system.retrieval_loop.candidates import (
    candidate_centered_video_input,
    candidate_id,
)
from application.system.secondStageEvaluator.prompt import (
    SECOND_STAGE_EVALUATOR_CONTEXT_PROMPT,
    SECOND_STAGE_EVALUATOR_SYSTEM_PROMPT,
)
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.executor import run
from application.system.vlm.prompt_trace import prompt_trace
from application.tasks.models import Dataset, TaskType
from application.types.retrieval.candidate import Candidate
from application.types.retrieval.queries import LoadVideoSummary
from application.types.second_stage_evaluator import (
    SecondStageEvaluationRequest,
    SecondStageEvaluationResult,
    SecondStageVideoClip,
)
from application.types.vlm.prompt import LlmAgent, Prompt, PromptContext


VlmRunner = Callable[[str, Prompt], Awaitable[str]]
CandidateResolver = Callable[[str, Dataset], Awaitable[Candidate]]
ClipResolver = Callable[[Candidate, Dataset], SecondStageVideoClip]


class SecondStageEvaluatorSettings:
    """Fixed media window and generation settings for the verifier."""

    def __init__(
        self,
        *,
        max_tokens: int = 1000,
        seconds_before: float = 2.5,
        seconds_after: float = 2.5,
        fps: int = 5,
        max_attempts: int = 2,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        if seconds_before < 0 or seconds_after < 0:
            raise ValueError("Context durations must not be negative.")
        if seconds_before + seconds_after <= 0:
            raise ValueError("Context duration must be greater than 0.")
        if fps <= 0:
            raise ValueError("fps must be greater than 0.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0.")
        self.max_tokens = max_tokens
        self.seconds_before = seconds_before
        self.seconds_after = seconds_after
        self.fps = fps
        self.max_attempts = max_attempts


class SecondStageEvaluator(ABC):
    """Verify one candidate keyframe against textual task input."""

    @abstractmethod
    async def evaluate(
        self,
        candidate_keyframe_id: str,
        task_input_text: str,
    ) -> SecondStageEvaluationResult:
        """Return a strict full-scene match decision with explicit evidence."""


class PromptSecondStageEvaluator(SecondStageEvaluator):
    """Fetches candidate context and asks a VLM for structured verification."""

    def __init__(
        self,
        model: str = "qwen",
        *,
        runner: VlmRunner = run,
        candidate_resolver: CandidateResolver | None = None,
        clip_resolver: ClipResolver | None = None,
        settings: SecondStageEvaluatorSettings | None = None,
        dataset: Dataset = Dataset.V3C,
        task_type: TaskType = TaskType.AVS,
    ) -> None:
        self.model = model
        self.runner = runner
        self.candidate_resolver = candidate_resolver or resolve_keyframe_candidate
        self.settings = settings or SecondStageEvaluatorSettings()
        self.clip_resolver = clip_resolver or self._materialize_clip
        self.dataset = dataset
        self.task_type = task_type

    async def evaluate(
        self,
        candidate_keyframe_id: str,
        task_input_text: str,
    ) -> SecondStageEvaluationResult:
        request = SecondStageEvaluationRequest(
            candidate_keyframe_id=candidate_keyframe_id,
            task_input_text=task_input_text,
        )
        candidate = await self.candidate_resolver(
            request.candidate_keyframe_id,
            self.dataset,
        )
        resolved_id = candidate_id(candidate)
        if resolved_id != request.candidate_keyframe_id:
            raise ValueError(
                "Candidate resolver returned a different keyframe: "
                f"expected {request.candidate_keyframe_id!r}, got {resolved_id!r}."
            )
        clip = await asyncio.to_thread(
            self.clip_resolver,
            candidate,
            self.dataset,
        )
        prompt = build_second_stage_evaluator_prompt(
            request,
            clip,
            settings=self.settings,
            dataset=self.dataset,
            task_type=self.task_type,
        )

        llm_calls: list[dict[str, Any]] = []
        last_error: ValueError | None = None
        for attempt in range(1, self.settings.max_attempts + 1):
            started = timer_start()
            response = await self.runner(self.model, prompt)
            duration_ms = elapsed_ms(started)
            call: dict[str, Any] = {
                "prompt": prompt_trace(prompt),
                "response": response,
                "model": self.model,
                "attempt": attempt,
                "duration_ms": duration_ms,
            }
            try:
                result = parse_second_stage_evaluator_output(
                    response,
                    request.candidate_keyframe_id,
                    clip,
                )
            except ValueError as error:
                call["error"] = str(error)
                llm_calls.append(call)
                last_error = error
                continue

            call["fully_matches"] = result.fully_matches
            llm_calls.append(call)
            return result.model_copy(
                update={
                    "metadata": {
                        "dataset": self.dataset.value,
                        "task_type": self.task_type.value,
                        "keyframe_timestamp_ms": clip.keyframe_timestamp_ms,
                        "clip_fps": clip.fps,
                        "llm_calls": llm_calls,
                    }
                }
            )

        raise ValueError(
            "Second-stage evaluator returned invalid structured output after "
            f"{self.settings.max_attempts} attempts."
        ) from last_error

    def _materialize_clip(
        self,
        candidate: Candidate,
        dataset: Dataset,
    ) -> SecondStageVideoClip:
        video, start_ms, end_ms = candidate_centered_video_input(
            candidate,
            dataset,
            before_seconds=self.settings.seconds_before,
            after_seconds=self.settings.seconds_after,
            frame_rate=self.settings.fps,
        )
        return SecondStageVideoClip(
            video=video,
            keyframe_timestamp_ms=candidate.time_keyframe,
            start_ms=start_ms,
            end_ms=end_ms,
            fps=self.settings.fps,
        )


async def resolve_keyframe_candidate(
    candidate_keyframe_id: str,
    dataset: Dataset,
) -> Candidate:
    """Fetch exact keyframe metadata from the data layer's video-summary endpoint."""

    normalized_id = candidate_keyframe_id.strip()
    query = LoadVideoSummary(
        dataset=dataset.value,
        itemId=normalized_id,
        candidateCount=500,
    )
    candidates = await fetch_candidates(
        get_endpoint_for_query(query),
        create_load_video_summary_payload(query),
    )
    try:
        return next(
            candidate
            for candidate in candidates
            if "_".join(candidate.id) == normalized_id
        )
    except StopIteration as error:
        raise ValueError(
            f"Data layer did not return keyframe {normalized_id!r}."
        ) from error


def build_second_stage_evaluator_prompt(
    request: SecondStageEvaluationRequest,
    clip: SecondStageVideoClip,
    *,
    settings: SecondStageEvaluatorSettings | None = None,
    dataset: Dataset = Dataset.V3C,
    task_type: TaskType = TaskType.AVS,
) -> Prompt:
    """Build the strict evidence-accounting prompt for one candidate clip."""

    settings = settings or SecondStageEvaluatorSettings()
    return Prompt(
        system_prompt=SECOND_STAGE_EVALUATOR_SYSTEM_PROMPT,
        context_prompt=SECOND_STAGE_EVALUATOR_CONTEXT_PROMPT.format(
            duration_seconds=(clip.end_ms - clip.start_ms) / 1000,
            keyframe_offset_seconds=(
                clip.keyframe_timestamp_ms - clip.start_ms
            )
            / 1000,
        ),
        max_tokens=settings.max_tokens,
        task_text=request.task_input_text,
        context_images=[],
        context_videos=[clip.video],
        context=PromptContext(
            llm_agent=LlmAgent.SECOND_STAGE_EVALUATOR,
            dataset=dataset,
            task_type=task_type,
            video_sampling_fps=clip.fps,
        ),
    )


def parse_second_stage_evaluator_output(
    response: str,
    candidate_keyframe_id: str,
    clip: SecondStageVideoClip,
) -> SecondStageEvaluationResult:
    """Parse and validate the evaluator's exact JSON response contract."""

    try:
        payload = json.loads(response.strip())
    except json.JSONDecodeError as error:
        raise ValueError(
            "Second-stage evaluator must return exactly one JSON object."
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Second-stage evaluator must return exactly one JSON object.")

    expected_fields = {
        "scene_description",
        "matching_details",
        "non_matching_details",
        "uncertain_details",
        "fully_matches",
        "reason",
    }
    if set(payload) != expected_fields:
        missing = sorted(expected_fields - set(payload))
        unexpected = sorted(set(payload) - expected_fields)
        raise ValueError(
            "Second-stage evaluator returned incorrect fields: "
            f"missing={missing}, unexpected={unexpected}."
        )
    try:
        return SecondStageEvaluationResult(
            candidate_keyframe_id=candidate_keyframe_id,
            clip_start_ms=clip.start_ms,
            clip_end_ms=clip.end_ms,
            **payload,
        )
    except (TypeError, ValidationError) as error:
        raise ValueError(
            "Second-stage evaluator returned invalid field values."
        ) from error
