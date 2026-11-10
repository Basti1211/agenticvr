"""Final video-question-answering module for QA retrieval tasks.

The module is intentionally separate from planner VQA actions: planner VQA is an
information-gathering action, while this module produces the final answer after
the retrieval/evaluation pipeline has selected the most likely video.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from application.settings.settings import _qa_settings, retrieval_candidate_count
from application.system.task_input_access import remove_task_question
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.executor import run, run_parallel
from application.system.vlm.prompt_trace import prompt_trace
from application.types.vqa.models import (
    VQAProgressCallback,
    VQAResult,
    VQASegment,
    VQASegmentAttempt,
)
from application.types.vqa.state import (
    VQAAnswerContext,
    VQASceneAnswerState,
    VQASceneSelection,
)
from application.tasks.models import EncodedTask, TaskCreate
from application.types.retrieval.candidate import Candidate
from application.types.vlm.prompt import (
    ImageInput,
    LlmAgent,
    Prompt,
    PromptContext,
    VideoInput,
)

from application.system.prompts.template import render_prompt
from application.system.vqa.prompt import (
    SEGMENT_ANSWER_SYSTEM_PROMPT,
    SEGMENT_ANSWER_CONTEXT_PROMPT,
    ANSWER_VERIFICATION_SYSTEM_PROMPT,
    ANSWER_VERIFICATION_CONTEXT_PROMPT,
    ANSWER_SELECTION_SYSTEM_PROMPT,
    ANSWER_SELECTION_CONTEXT_PROMPT,
    MISSING_EVIDENCE_OBSERVATION_SYSTEM_PROMPT,
    MISSING_EVIDENCE_OBSERVATION_CONTEXT_PROMPT,
    MISSING_EVIDENCE_REASONING_SYSTEM_PROMPT,
    MISSING_EVIDENCE_REASONING_CONTEXT_PROMPT,
    SCENE_NOTES_SYSTEM_PROMPT,
    SCENE_NOTES_CONTEXT_PROMPT,
    SCENE_NOTES_ANSWER_SYSTEM_PROMPT,
    SCENE_NOTES_ANSWER_CONTEXT_PROMPT,
)

logger = logging.getLogger(__name__)

_PREVIOUS_VQA_SUBMISSIONS_HEADER = "Previous VQA answer submissions:"
_PREVIOUS_VQA_SCENE_NOTES_HEADER = "Previous VQA scene notes JSON:"
N_MODE_SCENE_LIMIT = 15


@dataclass(frozen=True)
class _PreviousVQAAnswerSubmission:
    answer: str
    status: str
    reason: str | None = None
    segment_index: int | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    source_candidate_id: str | None = None
    scene_notes: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "reason": self.reason,
            "segment_index": self.segment_index,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "source_candidate_id": self.source_candidate_id,
            "scene_notes": self.scene_notes,
        }

VlmRunner = Callable[[str, Prompt], Awaitable[str]]
VlmParallelRunner = Callable[[str, list[Prompt]], Awaitable[list[str]]]
IntraVideoRetriever = Callable[
    [str, str, TaskCreate, int],
    Awaitable[list[Candidate]],
]
VideoSegmentFactory = Callable[[VideoInput, float, float | None], VideoInput]
SourceVideoDurationResolver = Callable[[Any, Any], float]
SourceVideoSegmentResolver = Callable[[str, int, int, Any], VideoInput]
TaskReferenceVideoResolver = Callable[[TaskCreate], VideoInput | None]


class VQA(ABC):
    """Answer the original QA question from a selected candidate video."""

    @abstractmethod
    async def answer(
        self,
        task: TaskCreate,
        candidate: Any,
        *,
        encoded_task: EncodedTask | None = None,
        task_text: str | None = None,
        include_task_text: bool = True,
        task_images: list[ImageInput] | None = None,
        task_videos: list[VideoInput] | None = None,
        progress_callback: VQAProgressCallback | None = None,
        previous_vqa_results: Sequence[VQAResult] = (),
    ) -> VQAResult:
        """Return a verified answer or an evidence-based failure reason."""


class PromptVQA(VQA):
    """Evaluator-style VQA over short video segments."""

    def __init__(
        self,
        model: str = "qwen",
        runner: VlmRunner = run,
        parallel_runner: VlmParallelRunner | None = None,
        *,
        segment_duration_seconds: float = 10.0,
        segment_overlap_seconds: float = 2.0,
        max_tokens: int = 256,
        intra_video_retriever: IntraVideoRetriever | None = None,
        intra_video_candidate_count: int | None = None,
        max_scenes: int | None = None,
        n_mode_scene_limit: int = N_MODE_SCENE_LIMIT,
        source_video_duration_resolver: SourceVideoDurationResolver | None = None,
        source_video_segment_resolver: SourceVideoSegmentResolver | None = None,
        task_reference_video_resolver: TaskReferenceVideoResolver | None = None,
        segment_factory: VideoSegmentFactory | None = None,
    ) -> None:
        if segment_duration_seconds <= 0:
            raise ValueError("segment_duration_seconds must be greater than 0")
        if segment_overlap_seconds < 0:
            raise ValueError("segment_overlap_seconds must not be negative")
        if segment_overlap_seconds >= segment_duration_seconds:
            raise ValueError(
                "segment_overlap_seconds must be smaller than segment_duration_seconds"
            )
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if intra_video_candidate_count is not None and intra_video_candidate_count <= 0:
            raise ValueError("intra_video_candidate_count must be greater than 0")
        if max_scenes is not None and max_scenes <= 0:
            raise ValueError("max_scenes must be greater than 0")
        if n_mode_scene_limit <= 0:
            raise ValueError("n_mode_scene_limit must be greater than 0")

        self.model = model
        self.runner = runner
        self.parallel_runner = (
            parallel_runner
            if parallel_runner is not None
            else (run_parallel if runner is run else None)
        )
        self.segment_duration_seconds = segment_duration_seconds
        self.segment_overlap_seconds = segment_overlap_seconds
        self.max_tokens = max_tokens
        self.intra_video_retriever = (
            intra_video_retriever or _default_intra_video_retriever
        )
        qa_settings = _qa_settings()
        self.intra_video_candidate_count = int(
            intra_video_candidate_count or retrieval_candidate_count()
        )
        self.max_scenes = int(max_scenes or qa_settings.get("vqa_max_scenes", 8))
        self.n_mode_scene_limit = n_mode_scene_limit
        self.source_video_duration_resolver = (
            source_video_duration_resolver or _default_source_video_duration_resolver
        )
        self.source_video_segment_resolver = (
            source_video_segment_resolver or _default_source_video_segment_resolver
        )
        if task_reference_video_resolver is None:
            from application.system.retrieval_loop.candidates import (
                task_reference_video_input,
            )

            task_reference_video_resolver = task_reference_video_input
        self.task_reference_video_resolver = task_reference_video_resolver
        self.segment_factory = segment_factory or _default_segment_factory

    def _metadata(
        self,
        *,
        segment_count: int,
        source_video_id: str | None,
        source_candidate_id: str | None,
        temporal_mode: str | None,
        time_specific: bool,
        vqa_strategy: str | None,
        query_candidates: Sequence[Candidate] = (),
        scene_candidates: Sequence[Candidate] = (),
        inspected_scene_candidates: Sequence[Candidate] = (),
        retrieval_duration_ms: float | None = None,
        calls: Sequence[dict[str, Any]] = (),
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "segment_count": segment_count,
            "source_video_id": source_video_id,
            "source_candidate_id": source_candidate_id,
            "qa_temporal_mode": temporal_mode,
            "qa_time_specific": time_specific,
            "vqa_strategy": vqa_strategy,
            "retrieval_candidate_count": len(query_candidates),
            "scene_candidate_count": len(scene_candidates),
            "inspected_scene_candidate_count": len(inspected_scene_candidates),
            "query_candidate_ids": [
                _candidate_identifier(item) for item in query_candidates
            ],
            "scene_candidate_ids": [
                _candidate_identifier(item) for item in scene_candidates
            ],
            "inspected_scene_candidate_ids": [
                _candidate_identifier(item) for item in inspected_scene_candidates
            ],
            "scene_keys": _scene_keys_metadata(
                _candidate_scene_key(item) for item in scene_candidates
            ),
            "inspected_scene_keys": _scene_keys_metadata(
                _candidate_scene_key(item) for item in inspected_scene_candidates
            ),
            "llm_calls": calls if isinstance(calls, list) else list(calls),
        }
        if retrieval_duration_ms is not None:
            metadata["retrieval_duration_ms"] = retrieval_duration_ms
        if extra:
            metadata.update(extra)
        return metadata

    def _result(
        self,
        *,
        answerable: bool,
        answer: str | None = None,
        reason: str | None = None,
        observations: Sequence[VQASegmentAttempt] = (),
        metadata: dict[str, Any],
    ) -> VQAResult:
        return VQAResult(
            answerable=answerable,
            answer=answer,
            reason=reason,
            observations=tuple(observations),
            metadata=metadata,
        )

    async def answer(
        self,
        task: TaskCreate,
        candidate: Any,
        *,
        encoded_task: EncodedTask | None = None,
        task_text: str | None = None,
        include_task_text: bool = True,
        task_images: list[ImageInput] | None = None,
        task_videos: list[VideoInput] | None = None,
        progress_callback: VQAProgressCallback | None = None,
        previous_vqa_results: Sequence[VQAResult] = (),
    ) -> VQAResult:
        context = await self._answer_context(
            task=task,
            candidate=candidate,
            encoded_task=encoded_task,
            task_text=task_text,
            include_task_text=include_task_text,
            task_images=task_images,
            task_videos=task_videos,
        )

        if context.source_video_id is None:
            return self._result(
                answerable=False,
                reason=(
                    "The accepted image candidate does not contain a parent video id."
                ),
                metadata=self._metadata(
                    segment_count=0,
                    source_video_id=None,
                    source_candidate_id=context.source_candidate_id,
                    temporal_mode=context.temporal_mode,
                    time_specific=context.time_specific,
                    vqa_strategy=None,
                    extra={"retrieval_error": "missing_source_video_id"},
                ),
            )

        reused_scene_notes = _previous_vqa_scene_note_attempts(
            context.previous_answer_submissions
        )
        if (
            context.temporal_mode == "V"
            and reused_scene_notes
            and not _encoded_vqa_refresh_scene_notes(encoded_task)
        ):
            return await self._answer_from_reused_chronological_scene_notes(
                task=task,
                context=context,
                attempts=reused_scene_notes,
            )

        selection = await self._select_scene_candidates(
            task,
            candidate,
            context,
            encoded_task,
        )
        if isinstance(selection, VQAResult):
            return selection

        if context.temporal_mode == "N":
            return await self._answer_from_expanded_intra_video_scene_batches(
                task=task,
                context=context,
                initial_selection=selection,
                progress_callback=progress_callback,
                previous_vqa_results=previous_vqa_results,
            )

        if not selection.scene_candidates:
            return self._result(
                answerable=False,
                reason=(
                    "The intra-video question query did not return any new scene "
                    "from the matched source video to inspect."
                ),
                metadata=self._metadata(
                    segment_count=0,
                    source_video_id=context.source_video_id,
                    source_candidate_id=context.source_candidate_id,
                    temporal_mode=context.temporal_mode,
                    time_specific=context.time_specific,
                    vqa_strategy=selection.vqa_strategy,
                    query_candidates=selection.query_candidates,
                    scene_candidates=selection.scene_candidates,
                    retrieval_duration_ms=selection.retrieval_duration_ms,
                ),
            )

        if context.temporal_mode == "V":
            return await self._answer_from_chronological_scene_notes(
                task=task,
                question=context.question,
                source_video_id=context.source_video_id,
                source_candidate_id=context.source_candidate_id,
                scene_candidates=selection.scene_candidates,
                query_candidates=selection.query_candidates,
                retrieval_duration_ms=selection.retrieval_duration_ms,
                previous_answer_submissions=context.previous_answer_submissions,
            )

        return await self._answer_from_candidate_scene_batches(
            task=task,
            context=context,
            selection=selection,
            progress_callback=progress_callback,
        )

    async def _answer_from_reused_chronological_scene_notes(
        self,
        *,
        task: TaskCreate,
        context: VQAAnswerContext,
        attempts: Sequence[VQASegmentAttempt],
    ) -> VQAResult:
        calls: list[dict[str, Any]] = []
        metadata = self._metadata(
            segment_count=len(attempts),
            source_video_id=context.source_video_id,
            source_candidate_id=context.source_candidate_id,
            temporal_mode=context.temporal_mode,
            time_specific=context.time_specific,
            vqa_strategy="chronological_scene_notes",
            calls=calls,
            extra={"reused_scene_notes": True},
        )
        answer, answer_call = await self._answer_from_scene_notes(
            task=task,
            question=context.question,
            attempts=attempts,
            previous_answer_submissions=context.previous_answer_submissions,
            temporal_mode="V",
        )
        calls.append(answer_call)
        if answer is None:
            return self._result(
                answerable=False,
                reason=(
                    "The reused chronological scene notes did not contain enough "
                    "evidence to find a different answer."
                ),
                observations=tuple(attempts),
                metadata=metadata,
            )
        metadata["selected_segment_index"] = None
        metadata["verified_segment_index"] = None
        metadata["answered_iteration"] = None
        return self._result(
            answerable=True,
            answer=answer,
            observations=tuple(attempts),
            metadata=metadata,
        )

    async def _answer_context(
        self,
        *,
        task: TaskCreate,
        candidate: Any,
        encoded_task: EncodedTask | None,
        task_text: str | None,
        include_task_text: bool,
        task_images: list[ImageInput] | None,
        task_videos: list[VideoInput] | None,
    ) -> VQAAnswerContext:
        task, previous_answer_submissions = _task_without_vqa_answer_feedback(task)
        question = _question_text(task)
        encoded_text = encoded_task.text if encoded_task is not None else None
        visible_task_text = (
            remove_task_question(task_text or encoded_text or "") or None
            if include_task_text
            else None
        )
        if task_videos is None:
            reference_video = (
                await asyncio.to_thread(self.task_reference_video_resolver, task)
                if _has_task_video_input(task)
                else None
            )
        else:
            reference_video = task_videos[0] if task_videos else None

        temporal_mode = _encoded_qa_temporal_mode(encoded_task)
        return VQAAnswerContext(
            question=question,
            visible_task_text=visible_task_text,
            task_images=list(task_images or []),
            reference_video=reference_video,
            source_video_id=_source_video_id(candidate),
            source_candidate_id=_candidate_identifier(candidate),
            temporal_mode=temporal_mode,
            time_specific=temporal_mode == "T",
            previous_answer_submissions=[
                item.as_dict() for item in previous_answer_submissions
            ],
        )

    async def _select_scene_candidates(
        self,
        task: TaskCreate,
        candidate: Any,
        context: VQAAnswerContext,
        encoded_task: EncodedTask | None,
    ) -> VQASceneSelection | VQAResult:
        if context.source_video_id is None:
            raise ValueError("Cannot select VQA scenes without a source video id.")

        if context.temporal_mode in {"T", "V"}:
            return self._chronological_scene_selection(
                task,
                candidate,
                context,
                encoded_task,
            )
        return await self._intra_video_scene_selection(task, context)

    def _chronological_scene_selection(
        self,
        task: TaskCreate,
        candidate: Any,
        context: VQAAnswerContext,
        encoded_task: EncodedTask | None,
    ) -> VQASceneSelection | VQAResult:
        retrieval_started = timer_start()
        vqa_strategy = (
            "chronological_scene_notes"
            if context.temporal_mode == "V"
            else "chronological_scan"
        )
        try:
            duration_seconds = self.source_video_duration_resolver(
                candidate,
                task.dataset,
            )
            start_ms, end_ms = _encoded_qa_inspection_range(
                encoded_task,
                duration_seconds=duration_seconds,
            )
            scene_candidates = _chronological_scene_candidates(
                context.source_video_id,
                duration_seconds,
                scene_duration_seconds=self.segment_duration_seconds,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        except Exception as error:
            logger.exception("vqa.source_video_duration.failed")
            return self._result(
                answerable=False,
                reason=(
                    "The source video duration could not be loaded for "
                    f"chronological VQA: {error}"
                ),
                metadata=self._metadata(
                    segment_count=0,
                    source_video_id=context.source_video_id,
                    source_candidate_id=context.source_candidate_id,
                    temporal_mode=context.temporal_mode,
                    time_specific=context.time_specific,
                    vqa_strategy=vqa_strategy,
                    extra={"retrieval_error": str(error)},
                ),
            )
        return VQASceneSelection(
            query_candidates=[],
            scene_candidates=scene_candidates,
            retrieval_duration_ms=elapsed_ms(retrieval_started),
            vqa_strategy=vqa_strategy,
        )

    async def _intra_video_scene_selection(
        self,
        task: TaskCreate,
        context: VQAAnswerContext,
        candidate_count: int | None = None,
    ) -> VQASceneSelection | VQAResult:
        if context.source_video_id is None:
            raise ValueError("Cannot run intra-video VQA without a source video id.")

        retrieval_started = timer_start()
        requested_count = candidate_count or self.intra_video_candidate_count
        try:
            query_candidates = await self.intra_video_retriever(
                context.source_video_id,
                context.question,
                task,
                requested_count,
            )
        except Exception as error:
            logger.exception("vqa.intra_video_query.failed")
            return self._result(
                answerable=False,
                reason=(
                    "The intra-video question query could not be executed against "
                    f"the data layer: {error}"
                ),
                metadata=self._metadata(
                    segment_count=0,
                    source_video_id=context.source_video_id,
                    source_candidate_id=context.source_candidate_id,
                    temporal_mode=context.temporal_mode,
                    time_specific=False,
                    vqa_strategy="intra_video_query",
                    extra={"retrieval_error": str(error)},
                ),
            )
        return VQASceneSelection(
            query_candidates=query_candidates,
            scene_candidates=_top_unique_scene_candidates(
                query_candidates,
                excluded_candidate_id=context.source_candidate_id,
            ),
            retrieval_duration_ms=elapsed_ms(retrieval_started),
            vqa_strategy="intra_video_query",
        )

    async def _answer_from_expanded_intra_video_scene_batches(
        self,
        *,
        task: TaskCreate,
        context: VQAAnswerContext,
        initial_selection: VQASceneSelection,
        progress_callback: VQAProgressCallback | None = None,
        previous_vqa_results: Sequence[VQAResult] = (),
    ) -> VQAResult:
        if context.source_video_id is None:
            raise ValueError(
                "Cannot answer VQA scene batches without a source video id."
            )

        previous_attempts = _previous_n_mode_attempts(
            previous_vqa_results,
            source_video_id=context.source_video_id,
        )
        previous_seen_scene_keys = _previous_n_mode_seen_scene_keys(
            previous_vqa_results,
            source_video_id=context.source_video_id,
        )
        state = VQASceneAnswerState(
            segments=[],
            attempts=list(previous_attempts),
            inspected_scene_candidates=[],
            calls=[],
            previous_segment_count=len(previous_attempts),
            previous_inspected_scene_keys=set(previous_seen_scene_keys),
            scene_limit=self.n_mode_scene_limit,
        )
        seen_scene_keys = set(previous_seen_scene_keys)
        all_scene_candidates: list[Candidate] = []
        selection = initial_selection
        retrieval_duration_ms = selection.retrieval_duration_ms
        requested_count = self.intra_video_candidate_count
        previous_query_count = 0
        iteration_index = 1

        reused_result = self._answer_from_previous_n_mode_observations(
            context=context,
            selection=selection,
            state=state,
        )
        if reused_result is not None:
            return reused_result

        reinspection_result = await self._answer_rejected_n_mode_segment_again(
            task=task,
            context=context,
            selection=selection,
            state=state,
            iteration_index=iteration_index,
            progress_callback=progress_callback,
        )
        if reinspection_result is not None:
            return reinspection_result
        if state.calls:
            iteration_index += 1

        while len(seen_scene_keys) < self.n_mode_scene_limit:
            new_scene_candidates = _new_unique_scene_candidates(
                selection.query_candidates,
                seen_scene_keys=seen_scene_keys,
                excluded_candidate_id=context.source_candidate_id,
                limit=self.n_mode_scene_limit - len(seen_scene_keys),
            )
            if new_scene_candidates:
                all_scene_candidates.extend(new_scene_candidates)
                current_selection = VQASceneSelection(
                    query_candidates=selection.query_candidates,
                    scene_candidates=list(all_scene_candidates),
                    retrieval_duration_ms=retrieval_duration_ms,
                    vqa_strategy=selection.vqa_strategy,
                )
                for scene_batch in _scene_batches(
                    new_scene_candidates,
                    self.max_scenes,
                ):
                    batch_result = await self._answer_scene_batch(
                        task=task,
                        context=context,
                        selection=current_selection,
                        state=state,
                        scene_batch=scene_batch,
                        iteration_index=iteration_index,
                        progress_callback=progress_callback,
                    )
                    if batch_result is not None:
                        return batch_result
                    iteration_index += 1

            if len(seen_scene_keys) >= self.n_mode_scene_limit:
                break

            query_count = len(selection.query_candidates)
            if query_count < requested_count or query_count <= previous_query_count:
                break

            previous_query_count = query_count
            requested_count += self.intra_video_candidate_count
            next_selection = await self._intra_video_scene_selection(
                task,
                context,
                candidate_count=requested_count,
            )
            if isinstance(next_selection, VQAResult):
                return next_selection
            retrieval_duration_ms += next_selection.retrieval_duration_ms
            selection = next_selection

        final_selection = VQASceneSelection(
            query_candidates=selection.query_candidates,
            scene_candidates=all_scene_candidates,
            retrieval_duration_ms=retrieval_duration_ms,
            vqa_strategy=selection.vqa_strategy,
        )
        return await self._finalize_unanswered_scene_batches(
            task,
            context,
            final_selection,
            state,
            generate_failure_reason=False,
        )

    def _answer_from_previous_n_mode_observations(
        self,
        *,
        context: VQAAnswerContext,
        selection: VQASceneSelection,
        state: VQASceneAnswerState,
    ) -> VQAResult | None:
        proposed_attempts = _new_proposed_attempts(
            state.attempts,
            context.previous_answer_submissions,
        )
        if not proposed_attempts:
            return None

        return self._answerable_scene_result(
            context,
            selection,
            state,
            [],
            proposed_attempts[0],
            0,
            extra={"reused_previous_n_mode_answer": True},
        )

    async def _answer_rejected_n_mode_segment_again(
        self,
        *,
        task: TaskCreate,
        context: VQAAnswerContext,
        selection: VQASceneSelection,
        state: VQASceneAnswerState,
        iteration_index: int,
        progress_callback: VQAProgressCallback | None = None,
    ) -> VQAResult | None:
        submission = _latest_reinspectable_n_mode_submission(
            context.previous_answer_submissions
        )
        if submission is None or context.source_video_id is None:
            return None

        start_seconds = _optional_float(submission.get("start_seconds"))
        end_seconds = _optional_float(submission.get("end_seconds"))
        if (
            start_seconds is None
            or end_seconds is None
            or end_seconds <= start_seconds
        ):
            return None

        start_ms = int(round(start_seconds * 1000))
        end_ms = int(round(end_seconds * 1000))
        video = self.source_video_segment_resolver(
            context.source_video_id,
            start_ms,
            end_ms,
            task.dataset,
        )
        source_candidate_id = _optional_str(submission.get("source_candidate_id"))
        segment = VQASegment(
            video=video,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            source_candidate_id=(
                source_candidate_id
                or f"{context.source_video_id}:{start_ms}-{end_ms}"
            ),
        )

        state.segments.append(segment)
        attempts, answer_calls = await self._answer_segments(
            task=task,
            question=context.question,
            task_text=context.visible_task_text,
            segments=[segment],
            segment_index_offset=_max_attempt_index(state.attempts),
            iteration_index=iteration_index,
            reference_video=context.reference_video,
            task_images=context.task_images,
            previous_answer_submissions=context.previous_answer_submissions,
            temporal_mode=context.temporal_mode,
        )
        state.calls.extend(answer_calls)
        extra = {
            "completed_iteration": iteration_index,
            "reinspected_rejected_n_mode_segment": True,
        }
        self._emit_progress(
            callback=progress_callback,
            context=context,
            selection=selection,
            state=state,
            observations=tuple([*state.attempts, *attempts]),
            extra=extra,
        )

        result = await self._result_from_batch_attempts(
            task=task,
            context=context,
            selection=selection,
            state=state,
            segments=[segment],
            attempts=attempts,
            iteration_index=iteration_index,
        )
        state.attempts.extend(attempts)
        return result

    async def _answer_from_candidate_scene_batches(
        self,
        *,
        task: TaskCreate,
        context: VQAAnswerContext,
        selection: VQASceneSelection,
        progress_callback: VQAProgressCallback | None = None,
    ) -> VQAResult:
        if context.source_video_id is None:
            raise ValueError(
                "Cannot answer VQA scene batches without a source video id."
            )

        state = VQASceneAnswerState(
            segments=[],
            attempts=[],
            inspected_scene_candidates=[],
            calls=[],
        )
        for iteration_index, scene_batch in enumerate(
            _scene_batches(selection.scene_candidates, self.max_scenes),
            start=1,
        ):
            batch_result = await self._answer_scene_batch(
                task=task,
                context=context,
                selection=selection,
                state=state,
                scene_batch=scene_batch,
                iteration_index=iteration_index,
                progress_callback=progress_callback,
            )
            if batch_result is not None:
                return batch_result

        return await self._finalize_unanswered_scene_batches(
            task,
            context,
            selection,
            state,
            generate_failure_reason=context.temporal_mode != "T",
        )

    async def _answer_scene_batch(
        self,
        *,
        task: TaskCreate,
        context: VQAAnswerContext,
        selection: VQASceneSelection,
        state: VQASceneAnswerState,
        scene_batch: Sequence[Candidate],
        iteration_index: int,
        progress_callback: VQAProgressCallback | None = None,
    ) -> VQAResult | None:
        if context.source_video_id is None:
            raise ValueError("Cannot resolve VQA segments without a source video id.")

        try:
            segments = await self._segments(task, context.source_video_id, scene_batch)
        except Exception as error:
            logger.exception("vqa.data_layer_video_resolution.failed")
            return self._result(
                answerable=False,
                reason=(
                    "The intra-video query scenes could not be loaded from the "
                    f"data layer: {error}"
                ),
                observations=tuple(state.attempts),
                metadata=self._metadata(
                    segment_count=_state_segment_count(state),
                    source_video_id=context.source_video_id,
                    source_candidate_id=context.source_candidate_id,
                    temporal_mode=context.temporal_mode,
                    time_specific=context.time_specific,
                    vqa_strategy=selection.vqa_strategy,
                    query_candidates=selection.query_candidates,
                    scene_candidates=selection.scene_candidates,
                    inspected_scene_candidates=state.inspected_scene_candidates,
                    retrieval_duration_ms=selection.retrieval_duration_ms,
                    calls=state.calls,
                    extra=_state_metadata_extra(
                        state,
                        {"source_video_resolution_error": str(error)},
                    ),
                ),
            )

        if not segments:
            return None

        state.inspected_scene_candidates.extend(scene_batch)
        state.segments.extend(segments)
        attempts, answer_calls = await self._answer_segments(
            task=task,
            question=context.question,
            task_text=context.visible_task_text,
            segments=segments,
            segment_index_offset=len(state.attempts),
            iteration_index=iteration_index,
            reference_video=context.reference_video,
            task_images=context.task_images,
            previous_answer_submissions=context.previous_answer_submissions,
            temporal_mode=context.temporal_mode,
        )
        state.calls.extend(answer_calls)
        self._emit_progress(
            callback=progress_callback,
            context=context,
            selection=selection,
            state=state,
            observations=tuple([*state.attempts, *attempts]),
            extra={"completed_iteration": iteration_index},
        )

        result = await self._result_from_batch_attempts(
            task=task,
            context=context,
            selection=selection,
            state=state,
            segments=segments,
            attempts=attempts,
            iteration_index=iteration_index,
        )
        state.attempts.extend(attempts)
        return result

    async def _result_from_batch_attempts(
        self,
        *,
        task: TaskCreate,
        context: VQAAnswerContext,
        selection: VQASceneSelection,
        state: VQASceneAnswerState,
        segments: Sequence[VQASegment],
        attempts: list[VQASegmentAttempt],
        iteration_index: int,
    ) -> VQAResult | None:
        proposed_attempts = _new_proposed_attempts(
            attempts,
            context.previous_answer_submissions,
        )
        if context.time_specific and proposed_attempts:
            return self._answerable_scene_result(
                context,
                selection,
                state,
                attempts,
                proposed_attempts[0],
                iteration_index,
            )
        if len(proposed_attempts) == 1:
            return self._answerable_scene_result(
                context,
                selection,
                state,
                attempts,
                proposed_attempts[0],
                iteration_index,
            )
        if len(proposed_attempts) <= 1:
            return None

        proposed_attempt_indexes = {attempt.index for attempt in proposed_attempts}
        attempts, verification_calls = await self._verify_proposed_answers(
            task=task,
            question=context.question,
            task_text=context.visible_task_text,
            segments=segments,
            attempts=attempts,
            proposed_attempt_indexes=proposed_attempt_indexes,
            reference_video=context.reference_video,
            task_images=context.task_images,
        )
        state.calls.extend(verification_calls)
        verified_attempts = [
            attempt for attempt in attempts if attempt.verified is True
        ]
        if not verified_attempts:
            return None

        selected, selection_call = await self._select_verified_answer(
            task=task,
            question=context.question,
            task_text=context.visible_task_text,
            attempts=verified_attempts,
            task_images=context.task_images,
            reference_video=context.reference_video,
        )
        if selection_call is not None:
            state.calls.append(selection_call)
        return self._answerable_scene_result(
            context,
            selection,
            state,
            attempts,
            selected,
            iteration_index,
            verified=True,
        )

    def _answerable_scene_result(
        self,
        context: VQAAnswerContext,
        selection: VQASceneSelection,
        state: VQASceneAnswerState,
        attempts: Sequence[VQASegmentAttempt],
        selected: VQASegmentAttempt,
        iteration_index: int,
        *,
        verified: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> VQAResult:
        result_extra = {
            "selected_segment_index": selected.index,
            "verified_segment_index": selected.index if verified else None,
            "answered_iteration": iteration_index,
        }
        if extra:
            result_extra.update(extra)

        return self._result(
            answerable=True,
            answer=selected.answer,
            observations=tuple([*state.attempts, *attempts]),
            metadata=self._metadata(
                segment_count=_state_segment_count(state),
                source_video_id=context.source_video_id,
                source_candidate_id=context.source_candidate_id,
                temporal_mode=context.temporal_mode,
                time_specific=context.time_specific,
                vqa_strategy=selection.vqa_strategy,
                query_candidates=selection.query_candidates,
                scene_candidates=selection.scene_candidates,
                inspected_scene_candidates=state.inspected_scene_candidates,
                retrieval_duration_ms=selection.retrieval_duration_ms,
                calls=state.calls,
                extra=_state_metadata_extra(state, result_extra),
            ),
        )

    async def _finalize_unanswered_scene_batches(
        self,
        task: TaskCreate,
        context: VQAAnswerContext,
        selection: VQASceneSelection,
        state: VQASceneAnswerState,
        *,
        generate_failure_reason: bool = True,
    ) -> VQAResult:
        if _state_segment_count(state) == 0:
            return self._result(
                answerable=False,
                reason=(
                    "The intra-video question query did not return any valid scene "
                    "segment to inspect."
                ),
                metadata=self._metadata(
                    segment_count=0,
                    source_video_id=context.source_video_id,
                    source_candidate_id=context.source_candidate_id,
                    temporal_mode=context.temporal_mode,
                    time_specific=context.time_specific,
                    vqa_strategy=selection.vqa_strategy,
                    query_candidates=selection.query_candidates,
                    scene_candidates=selection.scene_candidates,
                    retrieval_duration_ms=selection.retrieval_duration_ms,
                    calls=state.calls,
                ),
            )

        if generate_failure_reason:
            reason, failure_calls = await self._generate_failure_reason(
                task=task,
                question=context.question,
                task_text=context.visible_task_text,
                segments=state.segments,
                attempts=state.attempts,
                reference_video=context.reference_video,
                task_images=context.task_images,
            )
            state.calls.extend(failure_calls)
        else:
            reason = "The inspected scenes did not contain an answer."

        return self._result(
            answerable=False,
            reason=reason,
            observations=tuple(state.attempts),
            metadata=self._metadata(
                segment_count=_state_segment_count(state),
                source_video_id=context.source_video_id,
                source_candidate_id=context.source_candidate_id,
                temporal_mode=context.temporal_mode,
                time_specific=context.time_specific,
                vqa_strategy=selection.vqa_strategy,
                query_candidates=selection.query_candidates,
                scene_candidates=selection.scene_candidates,
                inspected_scene_candidates=state.inspected_scene_candidates,
                retrieval_duration_ms=selection.retrieval_duration_ms,
                calls=state.calls,
                extra=_state_metadata_extra(state),
            ),
        )

    def _emit_progress(
        self,
        *,
        callback: VQAProgressCallback | None,
        context: VQAAnswerContext,
        selection: VQASceneSelection,
        state: VQASceneAnswerState,
        observations: Sequence[VQASegmentAttempt],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if callback is None:
            return

        callback(
            self._result(
                answerable=False,
                reason=None,
                observations=tuple(observations),
                metadata=self._metadata(
                    segment_count=_state_segment_count(state),
                    source_video_id=context.source_video_id,
                    source_candidate_id=context.source_candidate_id,
                    temporal_mode=context.temporal_mode,
                    time_specific=context.time_specific,
                    vqa_strategy=selection.vqa_strategy,
                    query_candidates=selection.query_candidates,
                    scene_candidates=selection.scene_candidates,
                    inspected_scene_candidates=state.inspected_scene_candidates,
                    retrieval_duration_ms=selection.retrieval_duration_ms,
                    calls=state.calls,
                    extra=_state_metadata_extra(state, extra),
                ),
            )
        )

    async def _answer_from_chronological_scene_notes(
        self,
        *,
        task: TaskCreate,
        question: str,
        source_video_id: str,
        source_candidate_id: str | None,
        scene_candidates: Sequence[Candidate],
        query_candidates: Sequence[Candidate],
        retrieval_duration_ms: float,
        previous_answer_submissions: Sequence[dict[str, Any]] = (),
    ) -> VQAResult:
        calls: list[dict[str, Any]] = []
        all_segments: list[VQASegment] = []
        all_attempts: list[VQASegmentAttempt] = []
        inspected_scene_candidates: list[Candidate] = []

        for iteration_index, scene_batch in enumerate(
            _scene_batches(scene_candidates, self.max_scenes),
            start=1,
        ):
            try:
                segments = await self._segments(
                    task,
                    source_video_id,
                    scene_batch,
                )
            except Exception as error:
                logger.exception("vqa.scene_notes_segment_resolution.failed")
                return self._result(
                    answerable=False,
                    reason=(
                        "The chronological video scenes could not be loaded from "
                        f"the data layer: {error}"
                    ),
                    observations=tuple(all_attempts),
                    metadata=self._metadata(
                        segment_count=len(all_segments),
                        source_video_id=source_video_id,
                        source_candidate_id=source_candidate_id,
                        temporal_mode="V",
                        time_specific=False,
                        vqa_strategy="chronological_scene_notes",
                        query_candidates=query_candidates,
                        scene_candidates=scene_candidates,
                        inspected_scene_candidates=inspected_scene_candidates,
                        retrieval_duration_ms=retrieval_duration_ms,
                        calls=calls,
                        extra={"source_video_resolution_error": str(error)},
                    ),
                )

            if not segments:
                continue

            inspected_scene_candidates.extend(scene_batch)
            all_segments.extend(segments)
            attempts, observation_calls = await self._collect_scene_notes(
                task=task,
                question=question,
                segments=segments,
                segment_index_offset=len(all_attempts),
                iteration_index=iteration_index,
                previous_answer_submissions=previous_answer_submissions,
                temporal_mode="V",
            )
            all_attempts.extend(attempts)
            calls.extend(observation_calls)

        metadata = self._metadata(
            segment_count=len(all_segments),
            source_video_id=source_video_id,
            source_candidate_id=source_candidate_id,
            temporal_mode="V",
            time_specific=False,
            vqa_strategy="chronological_scene_notes",
            query_candidates=query_candidates,
            scene_candidates=scene_candidates,
            inspected_scene_candidates=inspected_scene_candidates,
            retrieval_duration_ms=retrieval_duration_ms,
            calls=calls,
        )

        scene_notes = [
            attempt for attempt in all_attempts if attempt.answer is not None
        ]
        if not all_segments:
            return self._result(
                answerable=False,
                reason=(
                    "The chronological video scan did not return any valid scene "
                    "segment to inspect."
                ),
                metadata=metadata,
            )
        if not scene_notes:
            return self._result(
                answerable=False,
                reason=(
                    "The chronological video scan did not find any notes relevant "
                    "to the question."
                ),
                observations=tuple(all_attempts),
                metadata=metadata,
            )

        answer, answer_call = await self._answer_from_scene_notes(
            task=task,
            question=question,
            attempts=scene_notes,
            previous_answer_submissions=previous_answer_submissions,
            temporal_mode="V",
        )
        calls.append(answer_call)

        if answer is None:
            return self._result(
                answerable=False,
                reason=(
                    "The collected chronological scene notes did not contain "
                    "enough evidence to answer the question."
                ),
                observations=tuple(all_attempts),
                metadata=metadata,
            )

        metadata["selected_segment_index"] = None
        metadata["verified_segment_index"] = None
        metadata["answered_iteration"] = None
        return self._result(
            answerable=True,
            answer=answer,
            observations=tuple(all_attempts),
            metadata=metadata,
        )

    async def _answer_segments(
        self,
        *,
        task: TaskCreate,
        question: str,
        task_text: str | None,
        segments: Sequence[VQASegment],
        segment_index_offset: int = 0,
        iteration_index: int = 1,
        reference_video: VideoInput | None,
        task_images: list[ImageInput],
        previous_answer_submissions: Sequence[dict[str, Any]] = (),
        temporal_mode: str | None = None,
    ) -> tuple[list[VQASegmentAttempt], list[dict[str, Any]]]:
        prompts = [
            _build_segment_answer_prompt(
                task=task,
                question=question,
                task_text=task_text,
                segment=segment,
                segment_index=segment_index_offset + index,
                segment_count=len(segments),
                iteration_index=iteration_index,
                max_tokens=self.max_tokens,
                reference_video=reference_video,
                task_images=task_images,
                previous_answer_submissions=previous_answer_submissions,
                temporal_mode=temporal_mode,
            )
            for index, segment in enumerate(segments, start=1)
        ]

        started = timer_start()
        responses = await self._run_parallel(prompts)
        duration_ms = elapsed_ms(started)

        attempts: list[VQASegmentAttempt] = []
        calls: list[dict[str, Any]] = []

        for local_index, (segment, prompt, response) in enumerate(
            zip(segments, prompts, responses, strict=True),
            start=1,
        ):
            index = segment_index_offset + local_index
            error: str | None = None
            try:
                answer = _parse_segment_answer(response)
            except ValueError as parse_error:
                answer = None
                error = str(parse_error)
            attempts.append(
                VQASegmentAttempt(
                    index=index,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    answer=answer,
                    raw_response=response.strip(),
                    verification_reason=error,
                )
            )
            call = {
                "phase": "segment_answer",
                "segment_index": index,
                "iteration": iteration_index,
                "source_candidate_id": segment.source_candidate_id,
                "prompt": prompt_trace(prompt),
                "response": response,
                "parsed_answer": answer,
                "model": self.model,
                "batch_duration_ms": duration_ms,
                "parallel_batch_size": len(prompts),
            }
            if error is not None:
                call["error"] = error
            calls.append(call)

        return attempts, calls

    async def _collect_scene_notes(
        self,
        *,
        task: TaskCreate,
        question: str,
        segments: Sequence[VQASegment],
        segment_index_offset: int,
        iteration_index: int,
        previous_answer_submissions: Sequence[dict[str, Any]] = (),
        temporal_mode: str | None = None,
    ) -> tuple[list[VQASegmentAttempt], list[dict[str, Any]]]:
        prompts = [
            _build_scene_notes_prompt(
                task=task,
                question=question,
                segment=segment,
                segment_index=segment_index_offset + index,
                segment_count=len(segments),
                iteration_index=iteration_index,
                max_tokens=self.max_tokens,
                previous_answer_submissions=previous_answer_submissions,
                temporal_mode=temporal_mode,
            )
            for index, segment in enumerate(segments, start=1)
        ]

        started = timer_start()
        responses = await self._run_parallel(prompts)
        duration_ms = elapsed_ms(started)

        attempts: list[VQASegmentAttempt] = []
        calls: list[dict[str, Any]] = []
        for local_index, (segment, prompt, response) in enumerate(
            zip(segments, prompts, responses, strict=True),
            start=1,
        ):
            index = segment_index_offset + local_index
            error: str | None = None
            try:
                notes = _parse_scene_notes(response)
            except ValueError as parse_error:
                notes = None
                error = str(parse_error)
            attempts.append(
                VQASegmentAttempt(
                    index=index,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    answer=notes,
                    raw_response=response.strip(),
                    verification_reason=error,
                )
            )
            call = {
                "phase": "scene_notes",
                "segment_index": index,
                "iteration": iteration_index,
                "source_candidate_id": segment.source_candidate_id,
                "prompt": prompt_trace(prompt),
                "response": response,
                "parsed_notes": notes,
                "model": self.model,
                "batch_duration_ms": duration_ms,
                "parallel_batch_size": len(prompts),
            }
            if error is not None:
                call["error"] = error
            calls.append(call)

        return attempts, calls

    async def _answer_from_scene_notes(
        self,
        *,
        task: TaskCreate,
        question: str,
        attempts: Sequence[VQASegmentAttempt],
        previous_answer_submissions: Sequence[dict[str, Any]] = (),
        temporal_mode: str | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        prompt = _build_scene_notes_answer_prompt(
            task=task,
            question=question,
            attempts=attempts,
            max_tokens=self.max_tokens,
            previous_answer_submissions=previous_answer_submissions,
            temporal_mode=temporal_mode,
        )
        started = timer_start()
        response = await self.runner(self.model, prompt)
        duration_ms = elapsed_ms(started)
        error: str | None = None
        try:
            answer = _parse_segment_answer(response)
        except ValueError as parse_error:
            answer = None
            error = str(parse_error)

        call = {
            "phase": "scene_notes_answer",
            "prompt": prompt_trace(prompt),
            "response": response,
            "parsed_answer": answer,
            "model": self.model,
            "duration_ms": duration_ms,
        }
        if error is not None:
            call["error"] = error
        return answer, call

    async def _verify_proposed_answers(
        self,
        *,
        task: TaskCreate,
        question: str,
        task_text: str | None,
        segments: Sequence[VQASegment],
        attempts: list[VQASegmentAttempt],
        proposed_attempt_indexes: set[int],
        reference_video: VideoInput | None,
        task_images: list[ImageInput],
    ) -> tuple[list[VQASegmentAttempt], list[dict[str, Any]]]:
        proposed = [
            (index, attempt)
            for index, attempt in enumerate(attempts)
            if attempt.answer is not None
            and attempt.index in proposed_attempt_indexes
        ]
        if not proposed:
            return attempts, []

        prompts = [
            _build_answer_verification_prompt(
                task=task,
                question=question,
                task_text=task_text,
                proposed_answer=attempt.answer or "",
                segment=segments[index],
                segment_index=attempt.index,
                max_tokens=self.max_tokens,
                reference_video=reference_video,
                task_images=task_images,
            )
            for index, attempt in proposed
        ]

        started = timer_start()
        responses = await self._run_parallel(prompts)
        duration_ms = elapsed_ms(started)

        calls: list[dict[str, Any]] = []
        updated = list(attempts)

        for (source_index, attempt), prompt, response in zip(
            proposed,
            prompts,
            responses,
            strict=True,
        ):
            try:
                verified, reason = _parse_verification(response)
                error: str | None = None
            except ValueError as parse_error:
                verified = False
                reason = str(parse_error)
                error = reason
            updated[source_index] = replace(
                attempt,
                verified=verified,
                verification_reason=reason,
            )
            call = {
                "phase": "answer_verification",
                "segment_index": attempt.index,
                "prompt": prompt_trace(prompt),
                "response": response,
                "verified": verified,
                "verification_reason": reason,
                "model": self.model,
                "batch_duration_ms": duration_ms,
                "parallel_batch_size": len(prompts),
            }
            if error is not None:
                call["error"] = error
            calls.append(call)

        return updated, calls

    async def _select_verified_answer(
        self,
        *,
        task: TaskCreate,
        question: str,
        task_text: str | None,
        attempts: Sequence[VQASegmentAttempt],
        task_images: list[ImageInput],
        reference_video: VideoInput | None,
    ) -> tuple[VQASegmentAttempt, dict[str, Any] | None]:
        if len(attempts) == 1:
            return attempts[0], None

        normalized_answers = {
            _normalize_answer(attempt.answer or "") for attempt in attempts
        }
        if len(normalized_answers) == 1:
            return attempts[0], None

        prompt = _build_answer_selection_prompt(
            task=task,
            question=question,
            task_text=task_text,
            attempts=attempts,
            max_tokens=self.max_tokens,
            task_images=task_images,
            reference_video=reference_video,
        )
        started = timer_start()
        response = await self.runner(self.model, prompt)
        duration_ms = elapsed_ms(started)
        selected_index = _parse_selected_segment(response, attempts)
        selected = next(
            attempt for attempt in attempts if attempt.index == selected_index
        )

        return selected, {
            "phase": "answer_selection",
            "prompt": prompt_trace(prompt),
            "response": response,
            "selected_segment_index": selected_index,
            "model": self.model,
            "duration_ms": duration_ms,
        }

    async def _generate_failure_reason(
        self,
        *,
        task: TaskCreate,
        question: str,
        task_text: str | None,
        segments: Sequence[VQASegment],
        attempts: Sequence[VQASegmentAttempt],
        reference_video: VideoInput | None,
        task_images: list[ImageInput],
    ) -> tuple[str, list[dict[str, Any]]]:
        observation_prompts = [
            _build_missing_evidence_observation_prompt(
                task=task,
                question=question,
                task_text=task_text,
                segment=segment,
                attempt=attempt,
                max_tokens=self.max_tokens,
                reference_video=reference_video,
                task_images=task_images,
            )
            for segment, attempt in zip(segments, attempts, strict=True)
        ]

        started = timer_start()
        responses = await self._run_parallel(observation_prompts)
        duration_ms = elapsed_ms(started)
        observations = [_normalize_text(response) for response in responses]

        calls: list[dict[str, Any]] = [
            {
                "phase": "failure_observation",
                "segment_index": attempt.index,
                "prompt": prompt_trace(prompt),
                "response": response,
                "model": self.model,
                "batch_duration_ms": duration_ms,
                "parallel_batch_size": len(observation_prompts),
            }
            for attempt, prompt, response in zip(
                attempts,
                observation_prompts,
                responses,
                strict=True,
            )
        ]

        reason_prompt = _build_missing_reason_prompt(
            task=task,
            question=question,
            task_text=task_text,
            attempts=attempts,
            observations=observations,
            max_tokens=self.max_tokens,
            task_images=task_images,
            reference_video=reference_video,
        )
        reason_started = timer_start()
        reason_response = await self.runner(self.model, reason_prompt)
        reason_duration_ms = elapsed_ms(reason_started)
        reason = _parse_missing_reason(reason_response)

        calls.append(
            {
                "phase": "failure_reason",
                "prompt": prompt_trace(reason_prompt),
                "response": reason_response,
                "model": self.model,
                "duration_ms": reason_duration_ms,
            }
        )
        return reason, calls

    async def _segments(
        self,
        task: TaskCreate,
        video_id: str,
        scene_candidates: Sequence[Candidate],
    ) -> list[VQASegment]:
        """Materialize top intra-video query scenes, splitting long scenes."""

        jobs: list[tuple[Candidate, int, int]] = []
        for candidate in scene_candidates:
            for start_ms, end_ms in _candidate_scene_segment_ranges_ms(
                candidate,
                segment_duration_seconds=self.segment_duration_seconds,
                overlap_seconds=self.segment_overlap_seconds,
            ):
                jobs.append((candidate, start_ms, end_ms))

        videos = [
            self.source_video_segment_resolver(
                video_id,
                start_ms,
                end_ms,
                task.dataset,
            )
            for _, start_ms, end_ms in jobs
        ]

        return [
            VQASegment(
                video=video,
                start_seconds=start_ms / 1000,
                end_seconds=end_ms / 1000,
                source_candidate_id=_candidate_identifier(candidate)
                or f"{video_id}:{start_ms}-{end_ms}",
            )
            for video, (candidate, start_ms, end_ms) in zip(
                videos,
                jobs,
                strict=True,
            )
        ]

    async def _run_parallel(self, prompts: list[Prompt]) -> list[str]:
        if not prompts:
            return []
        if self.parallel_runner is not None:
            return await self.parallel_runner(self.model, prompts)
        return await asyncio.gather(
            *(self.runner(self.model, prompt) for prompt in prompts)
        )


def _question_text(task: TaskCreate) -> str:
    task_input = getattr(task, "input", None)
    for source in (task_input, task):
        if source is None:
            continue
        for field in ("question", "description", "text"):
            value = (
                source.get(field)
                if isinstance(source, dict)
                else getattr(source, field, None)
            )
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("QA task does not contain an original question")


def _task_without_vqa_answer_feedback(
    task: TaskCreate,
) -> tuple[TaskCreate, list[_PreviousVQAAnswerSubmission]]:
    description = task.input.description
    if not description or _PREVIOUS_VQA_SUBMISSIONS_HEADER not in description:
        return task, []

    question_description, feedback_block = description.split(
        _PREVIOUS_VQA_SUBMISSIONS_HEADER,
        1,
    )
    previous_submissions = _parse_previous_vqa_answer_submissions(feedback_block)
    scene_notes = _parse_previous_vqa_scene_notes(feedback_block)
    if previous_submissions and scene_notes:
        previous_submissions[-1] = replace(
            previous_submissions[-1],
            scene_notes=scene_notes,
        )
    input_payload = task.input.model_copy(
        update={"description": question_description.strip() or None}
    )
    return task.model_copy(update={"input": input_payload}), previous_submissions


def _parse_previous_vqa_answer_submissions(
    feedback_block: str,
) -> list[_PreviousVQAAnswerSubmission]:
    submissions: list[_PreviousVQAAnswerSubmission] = []
    for raw_line in feedback_block.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        parsed = _parse_previous_vqa_answer_submission_line(line[2:])
        if parsed is not None:
            submissions.append(parsed)
    return submissions


def _parse_previous_vqa_scene_notes(
    feedback_block: str,
) -> list[dict[str, Any]] | None:
    if _PREVIOUS_VQA_SCENE_NOTES_HEADER not in feedback_block:
        return None

    _, notes_block = feedback_block.split(_PREVIOUS_VQA_SCENE_NOTES_HEADER, 1)
    notes_line = next(
        (line.strip() for line in notes_block.splitlines() if line.strip()),
        "",
    )
    if not notes_line:
        return None
    try:
        parsed = json.loads(notes_line)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    notes: list[dict[str, Any]] = []
    for item in parsed:
        note = _coerce_previous_scene_note(item)
        if note is not None:
            notes.append(note)
    return notes or None


def _coerce_previous_scene_note(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    notes = item.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        return None
    return {
        "segment_index": _optional_int(item.get("segment_index")),
        "start_seconds": _optional_float(item.get("start_seconds")),
        "end_seconds": _optional_float(item.get("end_seconds")),
        "notes": notes.strip(),
    }


def _parse_previous_vqa_answer_submission_line(
    line: str,
) -> _PreviousVQAAnswerSubmission | None:
    match = re.match(
        r"(?P<answer>.+): (?P<status>[a-z_]+)"
        r"(?: \((?P<reason>.*?)\))?"
        r"(?: \[(?P<context>.*)\])?$",
        line,
    )
    if match is None:
        return None

    answer = _literal_or_text(match.group("answer")).strip()
    if not answer:
        return None
    context = _parse_key_value_context(match.group("context") or "")
    return _PreviousVQAAnswerSubmission(
        answer=answer,
        status=match.group("status"),
        reason=match.group("reason"),
        segment_index=_optional_int(context.get("segment_index")),
        start_seconds=_optional_float(context.get("start_seconds")),
        end_seconds=_optional_float(context.get("end_seconds")),
        source_candidate_id=_optional_str(context.get("source_candidate_id")),
    )


def _literal_or_text(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    return parsed if isinstance(parsed, str) else value


def _parse_key_value_context(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in text.split(", "):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            result[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            result[key] = value
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _previous_vqa_scene_note_attempts(
    previous_answer_submissions: Sequence[dict[str, Any]],
) -> list[VQASegmentAttempt]:
    for submission in reversed(previous_answer_submissions):
        scene_notes = submission.get("scene_notes")
        if not isinstance(scene_notes, list):
            continue
        attempts: list[VQASegmentAttempt] = []
        for fallback_index, item in enumerate(scene_notes, start=1):
            if not isinstance(item, dict):
                continue
            notes = item.get("notes")
            if not isinstance(notes, str) or not notes.strip():
                continue
            attempts.append(
                VQASegmentAttempt(
                    index=_optional_int(item.get("segment_index")) or fallback_index,
                    start_seconds=_optional_float(item.get("start_seconds")) or 0.0,
                    end_seconds=_optional_float(item.get("end_seconds")),
                    answer=notes.strip(),
                    raw_response=f"S: {notes.strip()}",
                )
            )
        if attempts:
            return attempts
    return []


def _previous_n_mode_attempts(
    previous_vqa_results: Sequence[VQAResult],
    *,
    source_video_id: str,
) -> list[VQASegmentAttempt]:
    result = _latest_previous_n_mode_result(
        previous_vqa_results,
        source_video_id=source_video_id,
    )
    if result is None:
        return []
    return list(result.observations)


def _previous_n_mode_seen_scene_keys(
    previous_vqa_results: Sequence[VQAResult],
    *,
    source_video_id: str,
) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for result in _previous_n_mode_results(
        previous_vqa_results,
        source_video_id=source_video_id,
    ):
        metadata_keys = _metadata_scene_keys(result.metadata.get("inspected_scene_keys"))
        keys.update(metadata_keys)
        if metadata_keys:
            continue
        for observation in result.observations:
            start_ms = int(round(observation.start_seconds * 1000))
            end_seconds = observation.end_seconds
            if end_seconds is None:
                continue
            keys.add((source_video_id, start_ms, int(round(end_seconds * 1000))))
    return keys


def _latest_previous_n_mode_result(
    previous_vqa_results: Sequence[VQAResult],
    *,
    source_video_id: str,
) -> VQAResult | None:
    for result in reversed(previous_vqa_results):
        if _is_previous_n_mode_result(result, source_video_id=source_video_id):
            return result
    return None


def _previous_n_mode_results(
    previous_vqa_results: Sequence[VQAResult],
    *,
    source_video_id: str,
) -> list[VQAResult]:
    return [
        result
        for result in previous_vqa_results
        if _is_previous_n_mode_result(result, source_video_id=source_video_id)
    ]


def _is_previous_n_mode_result(
    result: VQAResult,
    *,
    source_video_id: str,
) -> bool:
    return (
        result.metadata.get("qa_temporal_mode") == "N"
        and result.metadata.get("source_video_id") == source_video_id
    )


def _metadata_scene_keys(value: Any) -> set[tuple[str, int, int]]:
    if not isinstance(value, list):
        return set()

    keys: set[tuple[str, int, int]] = set()
    for item in value:
        key = _metadata_scene_key(item)
        if key is not None:
            keys.add(key)
    return keys


def _metadata_scene_key(value: Any) -> tuple[str, int, int] | None:
    if isinstance(value, dict):
        video_id = value.get("video_id")
        start_ms = _optional_int(value.get("start_ms"))
        end_ms = _optional_int(value.get("end_ms"))
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        video_id = value[0]
        start_ms = _optional_int(value[1])
        end_ms = _optional_int(value[2])
    else:
        return None

    if not isinstance(video_id, str) or not video_id:
        return None
    if start_ms is None or end_ms is None:
        return None
    return video_id, start_ms, end_ms


def _has_task_video_input(task: TaskCreate) -> bool:
    media = task.input.media
    return task.input.task_video is not None or (
        media is not None and media.media_type == "video"
    )


async def _default_intra_video_retriever(
    video_id: str,
    question: str,
    task: TaskCreate,
    candidate_count: int,
) -> list[Candidate]:
    """Run a data-layer intra-video text query for the QA question."""

    from application.system.retrieval.endpoints import get_endpoint_for_query
    from application.system.retrieval.fetch import (
        create_payload,
        fetch_candidates_for_query,
    )
    from application.system.retrieval_loop.retrieval_adapter import (
        retrieval_dataset_name,
    )
    from application.types.retrieval.queries import IntraVideoQuery

    query = IntraVideoQuery(
        dataset=retrieval_dataset_name(task.dataset.value),
        videoId=video_id,
        query=question,
        candidateCount=candidate_count,
    )
    endpoint = get_endpoint_for_query(query)
    payload = create_payload(query)
    return await fetch_candidates_for_query(query, endpoint, payload)


def _state_segment_count(state: VQASceneAnswerState) -> int:
    return state.previous_segment_count + len(state.segments)


def _state_inspected_scene_keys(
    state: VQASceneAnswerState,
) -> set[tuple[str, int, int]]:
    keys = set(state.previous_inspected_scene_keys or set())
    keys.update(_candidate_scene_key(item) for item in state.inspected_scene_candidates)
    return keys


def _state_metadata_extra(
    state: VQASceneAnswerState,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(extra or {})
    inspected_scene_keys = _state_inspected_scene_keys(state)
    if inspected_scene_keys:
        result["inspected_scene_keys"] = _scene_keys_metadata(inspected_scene_keys)
    if state.previous_inspected_scene_keys:
        result["inspected_scene_candidate_count"] = len(inspected_scene_keys)
        result["previous_inspected_scene_candidate_count"] = len(
            state.previous_inspected_scene_keys
        )
    if state.previous_segment_count:
        result["previous_segment_count"] = state.previous_segment_count
    if state.scene_limit is not None:
        seen_count = len(inspected_scene_keys)
        result["n_mode_scene_limit"] = state.scene_limit
        result["n_mode_seen_scene_count"] = seen_count
        result["n_mode_scene_limit_reached"] = seen_count >= state.scene_limit
        result["n_mode_scene_slots_remaining"] = max(0, state.scene_limit - seen_count)
    return result


def _scene_keys_metadata(
    keys: Iterable[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    return [
        {"video_id": video_id, "start_ms": start_ms, "end_ms": end_ms}
        for video_id, start_ms, end_ms in sorted(keys)
    ]


def _top_unique_scene_candidates(
    candidates: Sequence[Candidate],
    *,
    excluded_candidate_id: str | None,
) -> list[Candidate]:
    """Keep top-ranked candidates while inspecting each scene only once."""

    result: list[Candidate] = []
    seen_scenes: set[tuple[str, int, int]] = set()

    for candidate in candidates:
        identifier = _candidate_identifier(candidate)
        if identifier is not None and identifier == excluded_candidate_id:
            continue

        scene_key = _candidate_scene_key(candidate)
        if scene_key in seen_scenes:
            continue
        seen_scenes.add(scene_key)
        result.append(candidate)

    return result


def _new_unique_scene_candidates(
    candidates: Sequence[Candidate],
    *,
    seen_scene_keys: set[tuple[str, int, int]],
    excluded_candidate_id: str | None,
    limit: int | None = None,
) -> list[Candidate]:
    result: list[Candidate] = []
    for candidate in candidates:
        if limit is not None and len(result) >= limit:
            break
        identifier = _candidate_identifier(candidate)
        if identifier is not None and identifier == excluded_candidate_id:
            continue

        scene_key = _candidate_scene_key(candidate)
        if scene_key in seen_scene_keys:
            continue
        seen_scene_keys.add(scene_key)
        result.append(candidate)
    return result


def _scene_batches(
    candidates: Sequence[Candidate],
    batch_size: int,
) -> list[list[Candidate]]:
    return [
        list(candidates[index : index + batch_size])
        for index in range(0, len(candidates), batch_size)
    ]


def _chronological_scene_candidates(
    video_id: str,
    duration_seconds: float,
    *,
    scene_duration_seconds: float,
    start_ms: int = 0,
    end_ms: int | None = None,
) -> list[Candidate]:
    if duration_seconds <= 0:
        raise ValueError("Source video duration must be greater than 0.")
    if scene_duration_seconds <= 0:
        raise ValueError("Chronological scene duration must be greater than 0.")

    duration_ms = max(1, int(round(duration_seconds * 1000)))
    scene_duration_ms = max(1, int(round(scene_duration_seconds * 1000)))
    if start_ms < 0:
        raise ValueError("Chronological inspection start must not be negative.")
    if start_ms >= duration_ms:
        raise ValueError("Chronological inspection start is outside the video.")
    resolved_end_ms = duration_ms if end_ms is None else min(end_ms, duration_ms)
    if resolved_end_ms <= start_ms:
        raise ValueError("Chronological inspection end must be after its start.")
    candidates: list[Candidate] = []
    scene_start_ms = start_ms
    index = 1
    while scene_start_ms < resolved_end_ms:
        scene_end_ms = min(scene_start_ms + scene_duration_ms, resolved_end_ms)
        candidates.append(
            Candidate(
                uri=(
                    f"vqa-chronological://{video_id}/"
                    f"{scene_start_ms}-{scene_end_ms}"
                ),
                rank=str(index),
                score=None,
                id=[video_id, f"chronological-{index}"],
                time_keyframe=scene_start_ms,
                time_scene_start=scene_start_ms,
                time_scene_end=scene_end_ms,
            )
        )
        scene_start_ms = scene_end_ms
        index += 1

    return candidates


def _candidate_scene_key(candidate: Candidate) -> tuple[str, int, int]:
    video_id = candidate.id[0] if candidate.id else ""
    return video_id, candidate.time_scene_start, candidate.time_scene_end


def _candidate_scene_segment_ranges_ms(
    candidate: Candidate,
    *,
    segment_duration_seconds: float,
    overlap_seconds: float,
) -> list[tuple[int, int]]:
    scene_start_ms = candidate.time_scene_start
    scene_end_ms = candidate.time_scene_end
    if scene_start_ms < 0:
        raise ValueError("Candidate scene start must not be negative.")
    if scene_end_ms <= scene_start_ms:
        raise ValueError("Candidate scene end must be greater than scene start.")

    duration_seconds = (scene_end_ms - scene_start_ms) / 1000
    ranges = _overlapping_segment_ranges(
        duration_seconds=duration_seconds,
        segment_duration_seconds=segment_duration_seconds,
        overlap_seconds=overlap_seconds,
    )
    return [
        (
            scene_start_ms + int(round(start_seconds * 1000)),
            scene_start_ms + int(round(end_seconds * 1000)),
        )
        for start_seconds, end_seconds in ranges
    ]


def _default_source_video_duration_resolver(
    candidate: Any,
    dataset: Any,
) -> float:
    """Resolve full-video duration using the retrieval-loop data layer."""

    from application.system.retrieval_loop.candidates import (
        candidate_source_video_duration_seconds,
    )

    return candidate_source_video_duration_seconds(candidate, dataset)


def _default_source_video_segment_resolver(
    video_id: str,
    start_ms: int,
    end_ms: int,
    dataset: Any,
) -> VideoInput:
    """Materialize a segment from the complete data-layer video."""

    from application.system.retrieval_loop.candidates import vqa_video_input

    return vqa_video_input(video_id, start_ms, end_ms, dataset)


def _overlapping_segment_ranges(
    *,
    duration_seconds: float,
    segment_duration_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    """Return complete-video ranges with overlap."""

    step_seconds = segment_duration_seconds - overlap_seconds
    ranges: list[tuple[float, float]] = []
    start_seconds = 0.0

    while start_seconds < duration_seconds:
        end_seconds = min(
            start_seconds + segment_duration_seconds,
            duration_seconds,
        )
        ranges.append((start_seconds, end_seconds))
        if end_seconds >= duration_seconds:
            break
        start_seconds += step_seconds

    return ranges


def _candidate_identifier(candidate: Any) -> str | None:
    """Return a stable candidate identifier for metadata."""

    identifier = getattr(candidate, "identifier", None)
    if isinstance(identifier, str) and identifier:
        return identifier

    raw_id = getattr(candidate, "id", None)
    if isinstance(raw_id, str) and raw_id:
        return raw_id
    if isinstance(raw_id, (list, tuple)) and raw_id:
        return "_".join(str(part) for part in raw_id)

    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        metadata_id = metadata.get("id")
        if isinstance(metadata_id, str) and metadata_id:
            return metadata_id
        if isinstance(metadata_id, (list, tuple)) and metadata_id:
            return "_".join(str(part) for part in metadata_id)

    return None


def _source_video_id(candidate: Any) -> str | None:
    """Extract the parent video id from a retrieved image candidate."""

    raw_id = getattr(candidate, "id", None)
    if isinstance(raw_id, (list, tuple)) and raw_id:
        video_id = raw_id[0]
        if isinstance(video_id, str) and video_id:
            return video_id

    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        metadata_id = metadata.get("id")
        if isinstance(metadata_id, (list, tuple)) and metadata_id:
            video_id = metadata_id[0]
            if isinstance(video_id, str) and video_id:
                return video_id

        for key in ("video_id", "videoId", "video_name"):
            video_id = metadata.get(key)
            if isinstance(video_id, str) and video_id:
                return video_id

    identifier = _candidate_identifier(candidate)
    if identifier and "_" in identifier:
        return identifier.rsplit("_", 1)[0]

    return None


def _encoded_qa_temporal_mode(encoded_task: EncodedTask | None) -> str:
    if encoded_task is None:
        return "V"

    mode = encoded_task.metadata.get("qa_temporal_mode")
    if isinstance(mode, str) and mode.upper() in {"N", "T", "V"}:
        return mode.upper()

    value = encoded_task.metadata.get("qa_time_specific")
    if isinstance(value, bool):
        return "T" if value else "N"
    return "V"


def _encoded_vqa_refresh_scene_notes(encoded_task: EncodedTask | None) -> bool:
    if encoded_task is None:
        return False
    return encoded_task.metadata.get("vqa_refresh_scene_notes") is True


def _encoded_qa_inspection_range(
    encoded_task: EncodedTask | None,
    *,
    duration_seconds: float,
) -> tuple[int, int | None]:
    if encoded_task is None:
        return 0, None

    start_ms = encoded_task.metadata.get("qa_inspection_start_ms", 0)
    end_ms = encoded_task.metadata.get("qa_inspection_end_ms")
    if not isinstance(start_ms, int) or isinstance(start_ms, bool) or start_ms < 0:
        raise ValueError(
            "QA inspection start timestamp must be a non-negative integer."
        )
    if end_ms is not None and (
        not isinstance(end_ms, int) or isinstance(end_ms, bool) or end_ms < 0
    ):
        raise ValueError(
            "QA inspection end timestamp must be a non-negative integer or END."
        )
    if end_ms is not None and start_ms >= end_ms:
        raise ValueError("QA inspection start timestamp must be before its end.")

    duration_ms = max(1, int(round(duration_seconds * 1000)))
    if start_ms >= duration_ms:
        raise ValueError("QA inspection start timestamp is outside the video.")
    return start_ms, end_ms


def _video_duration_seconds(
    candidate: Any,
    video: VideoInput,
) -> float | None:
    sources = (
        candidate,
        getattr(candidate, "metadata", None),
        video,
    )
    for source in sources:
        if source is None:
            continue

        for key in ("duration_seconds", "video_duration_seconds"):
            value = (
                source.get(key)
                if isinstance(source, dict)
                else getattr(source, key, None)
            )
            if _is_positive_number(value):
                return float(value)

        duration_ms = (
            source.get("duration_ms")
            if isinstance(source, dict)
            else getattr(source, "duration_ms", None)
        )
        if _is_positive_number(duration_ms):
            return float(duration_ms) / 1000.0

    return None


def _default_segment_factory(
    video: VideoInput,
    start_seconds: float,
    end_seconds: float | None,
) -> VideoInput:
    updates = {
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
    }

    if hasattr(video, "model_copy"):
        return video.model_copy(deep=True, update=updates)

    if hasattr(video, "copy"):
        try:
            return video.copy(update=updates)
        except TypeError:
            pass

    raise TypeError(
        "VideoInput cannot be temporally sliced. Pass a custom "
        "segment_factory that returns a clipped or ranged VideoInput."
    )


def _prompt_context(task: TaskCreate) -> PromptContext:
    return PromptContext(
        llm_agent=LlmAgent.VQA,
        dataset=task.dataset,
        task_type=task.task_type,
    )


def _answer_retry_context(
    previous_answer_submissions: Sequence[dict[str, Any]],
    *,
    temporal_mode: str | None,
    segment_index: int | None = None,
) -> str:
    submission = _latest_not_accepted_submission(previous_answer_submissions)
    if submission is None:
        return ""

    answer = str(submission.get("answer") or "").strip()
    if not answer:
        return ""

    status = str(submission.get("status") or "not_accepted")
    lines = [
        "",
        "PREVIOUS ANSWER RETRY:",
        _previous_answer_retry_status_line(answer, status),
        _previous_answer_retry_instruction(answer, status),
    ]

    target_index = _optional_int(submission.get("segment_index"))
    if temporal_mode in {"N", "T"} and target_index is not None:
        target_details = _segment_retry_details(submission)
        lines.append(
            "- For this N/T-mode retry, look again at the segment that produced "
            f"the previous answer: global_index {target_index}{target_details}."
        )
        if segment_index == target_index:
            lines.append(
                "- This attached segment is the previous answer segment. Inspect "
                "it especially carefully, but only return an answer that is "
                "different from the previous answer."
            )
        elif segment_index is not None:
            lines.append(
                "- This attached segment is not the previous answer segment. Use "
                "it only if it clearly provides a better supported answer."
            )

    return "\n".join(lines)


def _previous_answer_retry_status_line(answer: str, status: str) -> str:
    normalized_status = status.casefold()
    display_status = (
        "rejected"
        if normalized_status in {"rejected", "unavailable"}
        else normalized_status
    )
    return (
        f"- The previous submitted answer {answer!r} was classified as "
        f"{display_status} and was not accepted."
    )


def _previous_answer_retry_instruction(answer: str, status: str) -> str:
    return (
        f"- Treat {answer!r} as already rejected. Re-inspect the video evidence "
        "and return a different answer only if the segment supports it. Return "
        "F if there is no different supported answer."
    )


def _latest_not_accepted_submission(
    previous_answer_submissions: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    for item in reversed(previous_answer_submissions):
        if str(item.get("status") or "").casefold() != "accepted":
            return item
    return None


def _new_proposed_attempts(
    attempts: Sequence[VQASegmentAttempt],
    previous_answer_submissions: Sequence[dict[str, Any]],
) -> list[VQASegmentAttempt]:
    previous_answers = _previous_not_accepted_answers(previous_answer_submissions)
    return [
        attempt
        for attempt in attempts
        if attempt.answer is not None
        and _normalize_answer(attempt.answer) not in previous_answers
    ]


def _latest_reinspectable_n_mode_submission(
    previous_answer_submissions: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    for item in reversed(previous_answer_submissions):
        status = str(item.get("status") or "").casefold()
        if status not in {"rejected", "unavailable"}:
            continue
        start_seconds = _optional_float(item.get("start_seconds"))
        end_seconds = _optional_float(item.get("end_seconds"))
        if (
            start_seconds is None
            or end_seconds is None
            or end_seconds <= start_seconds
        ):
            continue
        return item
    return None


def _max_attempt_index(attempts: Sequence[VQASegmentAttempt]) -> int:
    return max((attempt.index for attempt in attempts), default=0)


def _previous_not_accepted_answers(
    previous_answer_submissions: Sequence[dict[str, Any]],
) -> set[str]:
    answers: set[str] = set()
    for item in previous_answer_submissions:
        if str(item.get("status") or "").casefold() == "accepted":
            continue
        answer = str(item.get("answer") or "")
        normalized = _normalize_answer(answer)
        if normalized:
            answers.add(normalized)
    return answers


def _segment_retry_details(submission: dict[str, Any]) -> str:
    details: list[str] = []
    source_candidate_id = submission.get("source_candidate_id")
    start_seconds = submission.get("start_seconds")
    end_seconds = submission.get("end_seconds")
    if isinstance(source_candidate_id, str) and source_candidate_id:
        details.append(f"source_candidate_id {source_candidate_id!r}")
    if start_seconds is not None:
        details.append(f"start_seconds {start_seconds}")
    if end_seconds is not None:
        details.append(f"end_seconds {end_seconds}")
    return f" ({', '.join(details)})" if details else ""


def _build_segment_answer_prompt(
    *,
    task: TaskCreate,
    question: str,
    task_text: str | None = None,
    segment: VQASegment,
    segment_index: int,
    segment_count: int,
    iteration_index: int,
    max_tokens: int,
    reference_video: VideoInput | None = None,
    task_images: list[ImageInput] | None = None,
    previous_answer_submissions: Sequence[dict[str, Any]] = (),
    temporal_mode: str | None = None,
) -> Prompt:
    return Prompt(
        system_prompt=render_prompt(
            SEGMENT_ANSWER_SYSTEM_PROMPT,
            task.task_type,
            task.dataset,
        ),
        context_prompt=render_prompt(
            SEGMENT_ANSWER_CONTEXT_PROMPT,
            task.task_type,
            task.dataset,
            QUESTION=question,
            VQA_ITERATION=iteration_index,
            SEGMENT_INDEX=segment_index,
            ITERATION_SEGMENT_COUNT=segment_count,
            START_SECONDS=segment.start_seconds,
            END_SECONDS=segment.end_seconds,
            SOURCE_CANDIDATE_ID=segment.source_candidate_id,
            ANSWER_RETRY_CONTEXT=_answer_retry_context(
                previous_answer_submissions,
                temporal_mode=temporal_mode,
                segment_index=segment_index,
            ),
        ),
        max_tokens=max_tokens,
        task_text=task_text,
        task_images=task_images or [],
        task_videos=[reference_video] if reference_video is not None else [],
        context_images=[],
        context_videos=[segment.video],
        context=_prompt_context(task),
    )


def _build_answer_verification_prompt(
    *,
    task: TaskCreate,
    question: str,
    task_text: str | None = None,
    proposed_answer: str,
    segment: VQASegment,
    segment_index: int,
    max_tokens: int,
    reference_video: VideoInput | None = None,
    task_images: list[ImageInput] | None = None,
) -> Prompt:
    return Prompt(
        system_prompt=render_prompt(
            ANSWER_VERIFICATION_SYSTEM_PROMPT,
            task.task_type,
            task.dataset,
        ),
        context_prompt=render_prompt(
            ANSWER_VERIFICATION_CONTEXT_PROMPT,
            task.task_type,
            task.dataset,
            QUESTION=question,
            PROPOSED_ANSWER=proposed_answer,
            SEGMENT_INDEX=segment_index,
            START_SECONDS=segment.start_seconds,
            END_SECONDS=segment.end_seconds,
            SOURCE_CANDIDATE_ID=segment.source_candidate_id,
        ),
        max_tokens=max_tokens,
        task_text=task_text,
        task_images=task_images or [],
        task_videos=[reference_video] if reference_video is not None else [],
        context_images=[],
        context_videos=[segment.video],
        context=_prompt_context(task),
    )


def _build_scene_notes_prompt(
    *,
    task: TaskCreate,
    question: str,
    segment: VQASegment,
    segment_index: int,
    segment_count: int,
    iteration_index: int,
    max_tokens: int,
    previous_answer_submissions: Sequence[dict[str, Any]] = (),
    temporal_mode: str | None = None,
) -> Prompt:
    return Prompt(
        system_prompt=render_prompt(
            SCENE_NOTES_SYSTEM_PROMPT,
            task.task_type,
            task.dataset,
        ),
        context_prompt=render_prompt(
            SCENE_NOTES_CONTEXT_PROMPT,
            task.task_type,
            task.dataset,
            QUESTION=question,
            VQA_ITERATION=iteration_index,
            SEGMENT_INDEX=segment_index,
            ITERATION_SEGMENT_COUNT=segment_count,
            START_SECONDS=segment.start_seconds,
            END_SECONDS=segment.end_seconds,
            SOURCE_CANDIDATE_ID=segment.source_candidate_id,
            ANSWER_RETRY_CONTEXT=_answer_retry_context(
                previous_answer_submissions,
                temporal_mode=temporal_mode,
                segment_index=segment_index,
            ),
        ),
        max_tokens=max_tokens,
        context_images=[],
        context_videos=[segment.video],
        context=_prompt_context(task),
    )


def _build_scene_notes_answer_prompt(
    *,
    task: TaskCreate,
    question: str,
    attempts: Sequence[VQASegmentAttempt],
    max_tokens: int,
    previous_answer_submissions: Sequence[dict[str, Any]] = (),
    temporal_mode: str | None = None,
) -> Prompt:
    scene_notes = [
        {
            "segment_index": attempt.index,
            "start_seconds": attempt.start_seconds,
            "end_seconds": attempt.end_seconds,
            "notes": attempt.answer,
        }
        for attempt in attempts
    ]
    return Prompt(
        system_prompt=render_prompt(
            SCENE_NOTES_ANSWER_SYSTEM_PROMPT,
            task.task_type,
            task.dataset,
        ),
        context_prompt=render_prompt(
            SCENE_NOTES_ANSWER_CONTEXT_PROMPT,
            task.task_type,
            task.dataset,
            QUESTION=question,
            ANSWER_RETRY_CONTEXT=_answer_retry_context(
                previous_answer_submissions,
                temporal_mode=temporal_mode,
            ),
            SCENE_NOTES=json.dumps(
                scene_notes,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
        max_tokens=max_tokens,
        context_images=[],
        context_videos=[],
        context=_prompt_context(task),
    )


def _build_answer_selection_prompt(
    *,
    task: TaskCreate,
    question: str,
    task_text: str | None = None,
    attempts: Sequence[VQASegmentAttempt],
    max_tokens: int,
    task_images: list[ImageInput] | None = None,
    reference_video: VideoInput | None = None,
) -> Prompt:
    candidates = [
        {
            "global_index": attempt.index,
            "start_seconds": attempt.start_seconds,
            "end_seconds": attempt.end_seconds,
            "answer": attempt.answer,
            "verification_reason": attempt.verification_reason,
        }
        for attempt in attempts
    ]
    return Prompt(
        system_prompt=render_prompt(
            ANSWER_SELECTION_SYSTEM_PROMPT,
            task.task_type,
            task.dataset,
        ),
        context_prompt=render_prompt(
            ANSWER_SELECTION_CONTEXT_PROMPT,
            task.task_type,
            task.dataset,
            QUESTION=question,
            VERIFIED_ANSWERS=json.dumps(
                candidates,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
        max_tokens=min(max_tokens, 32),
        task_text=task_text,
        task_images=task_images or [],
        task_videos=[reference_video] if reference_video is not None else [],
        context_images=[],
        context_videos=[],
        context=_prompt_context(task),
    )


def _build_missing_evidence_observation_prompt(
    *,
    task: TaskCreate,
    question: str,
    task_text: str | None = None,
    segment: VQASegment,
    attempt: VQASegmentAttempt,
    max_tokens: int,
    reference_video: VideoInput | None = None,
    task_images: list[ImageInput] | None = None,
) -> Prompt:
    return Prompt(
        system_prompt=render_prompt(
            MISSING_EVIDENCE_OBSERVATION_SYSTEM_PROMPT,
            task.task_type,
            task.dataset,
        ),
        context_prompt=render_prompt(
            MISSING_EVIDENCE_OBSERVATION_CONTEXT_PROMPT,
            task.task_type,
            task.dataset,
            QUESTION=question,
            SEGMENT_INDEX=attempt.index,
            START_SECONDS=segment.start_seconds,
            END_SECONDS=segment.end_seconds,
            SOURCE_CANDIDATE_ID=segment.source_candidate_id,
            SEGMENT_ANSWER_ATTEMPT=attempt.raw_response,
            VERIFICATION_REASON=attempt.verification_reason or "none",
        ),
        max_tokens=max_tokens,
        task_text=task_text,
        task_images=task_images or [],
        task_videos=[reference_video] if reference_video is not None else [],
        context_images=[],
        context_videos=[segment.video],
        context=_prompt_context(task),
    )


def _build_missing_reason_prompt(
    *,
    task: TaskCreate,
    question: str,
    task_text: str | None = None,
    attempts: Sequence[VQASegmentAttempt],
    observations: Sequence[str],
    max_tokens: int,
    task_images: list[ImageInput] | None = None,
    reference_video: VideoInput | None = None,
) -> Prompt:
    evidence = [
        {
            "start_seconds": attempt.start_seconds,
            "end_seconds": attempt.end_seconds,
            "answer_attempt": attempt.answer,
            "verification_reason": attempt.verification_reason,
            "observation": observation,
        }
        for attempt, observation in zip(attempts, observations, strict=True)
    ]
    return Prompt(
        system_prompt=render_prompt(
            MISSING_EVIDENCE_REASONING_SYSTEM_PROMPT,
            task.task_type,
            task.dataset,
        ),
        context_prompt=render_prompt(
            MISSING_EVIDENCE_REASONING_CONTEXT_PROMPT,
            task.task_type,
            task.dataset,
            QUESTION=question,
            SEGMENT_EVIDENCE=json.dumps(
                evidence,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
        max_tokens=max_tokens,
        task_text=task_text,
        task_images=task_images or [],
        task_videos=[reference_video] if reference_video is not None else [],
        context_images=[],
        context_videos=[],
        context=_prompt_context(task),
    )


def _parse_segment_answer(response: str) -> str | None:
    text = _normalize_text(response)

    if text.upper() == "F":
        return None

    match = re.fullmatch(r"A\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            "VQA segment response must be exactly 'F' or 'A: <answer>'. "
            f"Received: {response!r}"
        )

    answer = match.group(1).strip()
    if not answer:
        raise ValueError("VQA segment answer must not be empty")
    return answer


def _parse_scene_notes(response: str) -> str | None:
    text = _normalize_text(response)

    if text.upper() == "N":
        return None

    match = re.fullmatch(r"S\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            "VQA scene notes response must be exactly 'N' or "
            f"'S: <notes>'. Received: {response!r}"
        )

    notes = match.group(1).strip()
    if not notes:
        raise ValueError("VQA scene notes must not be empty")
    return notes


def _parse_verification(response: str) -> tuple[bool, str]:
    text = _normalize_text(response)
    match = re.fullmatch(r"([MF])\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            "VQA verification must be 'M: <reason>' or 'F: <reason>'. "
            f"Received: {response!r}"
        )

    return match.group(1).upper() == "M", match.group(2).strip()


def _parse_selected_segment(
    response: str,
    attempts: Sequence[VQASegmentAttempt],
) -> int:
    text = _normalize_text(response)
    match = re.search(r"\d+", text)
    if match is None:
        raise ValueError(
            "VQA answer selection must return a segment index. "
            f"Received: {response!r}"
        )

    selected = int(match.group(0))
    allowed = {attempt.index for attempt in attempts}
    if selected not in allowed:
        raise ValueError(f"Selected segment {selected} is not one of {sorted(allowed)}")
    return selected


def _parse_missing_reason(response: str) -> str:
    reason = _normalize_text(response)
    if reason.upper() == "F":
        raise ValueError("VQA missing-evidence reason must be descriptive")
    return reason


def _normalize_text(response: str) -> str:
    text = " ".join(
        line.strip() for line in response.strip().splitlines() if line.strip()
    )
    text = text.strip().strip("`").strip()
    if not text:
        raise ValueError("VQA response must not be empty")
    return text


def _normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer).strip().casefold().rstrip(".?!")


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
