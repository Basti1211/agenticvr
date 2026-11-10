"""Automated retrieval loop orchestration."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from application.settings.settings import (
    _data_layer_settings,
    _qa_settings,
    retrieval_candidate_count,
)
from application.system.answerer import (
    Answerer,
    PromptAnswerer,
)
from application.system.browser import Browser, PromptBrowser
from application.system.planner import Planner, configured_prompt_planner
from application.system.qa import PromptQuestionAnswerer, QuestionAnswerer
from application.system.retrieval.candidate_filters import unique_candidates
from application.system.retrieval_loop.action_handlers import ActionHandlers
from application.system.retrieval_loop.answerer_observer import AnswererObserver
from application.system.retrieval_loop.browser_pipeline import (
    BrowserInspection,
    BrowserPipeline,
)
from application.system.retrieval_loop.candidates import (
    CandidateImageResolver,
    CandidateVideoResolver,
    DirectAnswerResolver,
    QAVideoResolver,
    TaskReferenceVideoResolver,
    candidate_id,
    candidate_ids,
    candidate_image_input,
    direct_answer_candidate,
    qa_video_input,
    task_reference_video_input,
)
from application.system.retrieval_loop.history import (
    add_timing,
    jsonable,
    llm_history_entries,
    planner_direct_answer_rejected_history_step,
    planner_feedback_history_step,
)
from application.system.retrieval_loop.search_stages import (
    BrowserRetriever,
    Retriever,
    SearchStages,
)
from application.system.retrieval_loop.submission import verify_text_answer
from application.system.retrieval_loop.state import (
    IterationContext,
    IterationOutcome,
    RunContext,
)
from application.system.retrieval_loop.verification import VerificationPipeline
from application.system.task_encoder.encoder import TaskEncoder, VLMTaskEncoder
from application.system.task_input_access import (
    TaskInputAccessSettings,
    encoded_task_for_module,
    load_task_input_access_settings,
    task_description_for_module,
    task_images_for_module,
    task_videos_for_module,
)
from application.system.vqa import PromptVQA, VQA
from application.system.vqa_planner import VQAPlanner, VQAPlannerSession
from application.system.vlm.candidate_video_aliases import candidate_video_alias_scope
from application.system.timing import elapsed_ms, timer_start
from application.tasks.data_layer_paths import normalize_media_uri_path
from application.tasks.models import EncodedTask, TaskCreate, TaskType
from application.types.answerer.models import AnswererValidationStatus
from application.types.planner.models import (
    ImageQAAction,
    PlannerCondensedHistory,
    PlannerDecision,
    VideoQAAction,
)
from application.types.browser.models import BrowserReasoningOutput
from application.types.retrieval.queries import DirectAnswer, TextQuery
from application.types.retrieval_loop.models import (
    AVSMatch,
    CandidateSubmission,
    CandidateSubmissionStatus,
    RetrievalHistoryEntry,
    RetrievalLoopIteration,
    RetrievalLoopResult,
    RetrievalLoopStatus,
)
from application.types.retrieval_loop.evaluation import (
    CandidateEvaluationDecision,
    CandidateEvaluationResult,
)
from application.types.retrieval.candidate import Candidate
from application.types.vqa import VQAResult
from application.types.vqa_planner import VQAPlannerSessionResult


logger = logging.getLogger(__name__)
DEFAULT_MAX_CANDIDATES_PER_ITERATION = retrieval_candidate_count()


@dataclass(frozen=True)
class _EncodedTaskStep:
    encoded: EncodedTask
    duration_ms: float
    source: str


@dataclass
class _VQAAnswerFeedback:
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


@dataclass
class _VQAAttemptState:
    results: list[VQAResult]
    answer_feedback: list[_VQAAnswerFeedback]
    rejected_answer_reasons: list[str]
    nonaccepted_retry_pending: bool = False
    nonaccepted_retry_used: bool = False
    accepted_result: VQAResult | None = None


@dataclass(frozen=True)
class _LoopBounds:
    iteration_count: int
    candidate_limit: int


class RetrievalLoop:
    """Coordinate task encoding, planning, retrieval, and phase pipelines."""

    def __init__(
        self,
        *,
        encoder: TaskEncoder,
        planner: Planner,
        browser: Browser,
        answerer: Answerer | None = None,
        qa: QuestionAnswerer | None = None,
        vqa: VQA | None = None,
        vqa_planner: VQAPlanner | None = None,
        retriever: Retriever | None = None,
        browser_retriever: BrowserRetriever | None = None,
        image_resolver: CandidateImageResolver | None = None,
        browser_video_resolver: CandidateVideoResolver | None = None,
        direct_answer_resolver: DirectAnswerResolver | None = None,
        qa_image_resolver: CandidateImageResolver | None = None,
        qa_video_resolver: QAVideoResolver | None = None,
        task_reference_video_resolver: TaskReferenceVideoResolver | None = None,
        browser_video_followup: bool | None = None,
        task_input_access: TaskInputAccessSettings | None = None,
        max_iterations: int = 3,
        max_candidates_per_iteration: int = DEFAULT_MAX_CANDIDATES_PER_ITERATION,
        max_vqa_attempts: int = _qa_settings().get("max_vqa_attemps", 3),
        max_vqa_planner_steps: int = _qa_settings().get(
            "vqa_planner_max_steps",
            16,
        ),
        hide_avs_submission_ratings_from_planner: bool = False,
        description_hint_time_source: Callable[[], float] = timer_start,
    ) -> None:
        _validate_limit("max_iterations", max_iterations)
        _validate_limit(
            "max_candidates_per_iteration",
            max_candidates_per_iteration,
        )
        _validate_limit("max_vqa_attempts", max_vqa_attempts)
        _validate_limit("max_vqa_planner_steps", max_vqa_planner_steps)

        self.encoder = encoder
        self.planner = planner
        self.browser = browser
        self.answerer = answerer
        self.qa = qa or PromptQuestionAnswerer()
        self.vqa = vqa or PromptVQA()
        self.vqa_planner = vqa_planner
        self.retriever, self.browser_retriever = self._retrievers(
            retriever,
            browser_retriever,
        )
        self.image_resolver = image_resolver
        self.browser_video_resolver = browser_video_resolver
        self.direct_answer_resolver = direct_answer_resolver or direct_answer_candidate
        self.qa_image_resolver = (
            qa_image_resolver or image_resolver or candidate_image_input
        )
        self.qa_video_resolver = qa_video_resolver or qa_video_input
        self.task_reference_video_resolver = (
            task_reference_video_resolver or task_reference_video_input
        )
        if browser_video_followup is None:
            # Compatibility: an image resolver by itself has historically selected
            # the image-only path used by tests and custom callers.
            browser_video_followup = (
                browser_video_resolver is not None or image_resolver is None
            )
        self.browser_video_followup = browser_video_followup
        self.max_iterations = max_iterations
        self.max_candidates_per_iteration = max_candidates_per_iteration
        self.max_vqa_attempts = max_vqa_attempts
        self.max_vqa_planner_steps = max_vqa_planner_steps
        self.hide_avs_submission_ratings_from_planner = (
            hide_avs_submission_ratings_from_planner
        )
        self.description_hint_time_source = description_hint_time_source
        self.task_input_access = task_input_access or load_task_input_access_settings()
        self.vqa_planner_session = (
            VQAPlannerSession(
                planner=self.vqa_planner,
                qa=self.qa,
                vqa=self.vqa,
                video_resolver=self.qa_video_resolver,
                candidate_count=retrieval_candidate_count(),
                max_steps=self.max_vqa_planner_steps,
            )
            if self.vqa_planner is not None
            else None
        )

        self._configure_pipelines()

    def _retrievers(
        self,
        retriever: Retriever | None,
        browser_retriever: BrowserRetriever | None,
    ) -> tuple[Retriever, BrowserRetriever | None]:
        if retriever is not None:
            return retriever, browser_retriever

        from application.system.retrieval.fetch import (
            run_query,
            run_query_for_browser,
        )

        return run_query, browser_retriever or run_query_for_browser

    def _configure_pipelines(self) -> None:
        self.verification_pipeline = VerificationPipeline()
        self.browser_pipeline = BrowserPipeline(
            browser=self.browser,
            image_resolver=self.image_resolver,
            video_resolver=self.browser_video_resolver,
            video_followup=self.browser_video_followup,
            task_input_access=self.task_input_access.browser,
            task_reference_video_resolver=self.task_reference_video_resolver,
        )
        self.action_handlers = ActionHandlers(
            qa=self.qa,
            qa_image_resolver=self.qa_image_resolver,
            qa_video_resolver=self.qa_video_resolver,
            direct_answer_resolver=self.direct_answer_resolver,
            verification=self.verification_pipeline,
        )
        self.answerer_observer = AnswererObserver(
            self.answerer,
            task_input_access=self.task_input_access.answerer,
            task_reference_video_resolver=self.task_reference_video_resolver,
        )
        self.search_stages = SearchStages(
            planner=self.planner,
            retriever=self.retriever,
            browser_retriever=self.browser_retriever,
            planner_image_resolver=self.image_resolver,
            task_input_access=self.task_input_access.planner,
            task_reference_video_resolver=self.task_reference_video_resolver,
            hide_avs_submission_ratings_from_planner=(
                self.hide_avs_submission_ratings_from_planner
            ),
        )

    async def run(
        self,
        task: TaskCreate,
        *,
        encoded_task: EncodedTask | None = None,
        previous_result: RetrievalLoopResult | None = None,
        task_description_hints: list[str] | tuple[str, ...] | None = None,
        max_iterations: int | None = None,
        max_candidates_per_iteration: int | None = None,
        on_iteration_start: Callable[[int], None] | None = None,
        on_iteration_complete: Callable[[RetrievalLoopIteration], None] | None = None,
        on_iteration_progress: Callable[[RetrievalLoopIteration], None] | None = None,
    ) -> RetrievalLoopResult:
        """Run the retrieval loop until verified acceptance or exhaustion."""

        failure_start = timer_start()
        try:
            with candidate_video_alias_scope():
                return await self._run(
                    task,
                    encoded_task=encoded_task,
                    previous_result=previous_result,
                    task_description_hints=task_description_hints,
                    max_iterations=max_iterations,
                    max_candidates_per_iteration=max_candidates_per_iteration,
                    on_iteration_start=on_iteration_start,
                    on_iteration_complete=on_iteration_complete,
                    on_iteration_progress=on_iteration_progress,
                )
        except Exception:
            logger.exception(
                "retrieval_loop.failed dataset=%s task_type=%s total_ms=%.3f",
                task.dataset.value,
                task.task_type.value,
                elapsed_ms(failure_start),
            )
            raise

    async def _run(
        self,
        task: TaskCreate,
        *,
        encoded_task: EncodedTask | None = None,
        previous_result: RetrievalLoopResult | None = None,
        task_description_hints: list[str] | tuple[str, ...] | None = None,
        max_iterations: int | None = None,
        max_candidates_per_iteration: int | None = None,
        on_iteration_start: Callable[[int], None] | None = None,
        on_iteration_complete: Callable[[RetrievalLoopIteration], None] | None = None,
        on_iteration_progress: Callable[[RetrievalLoopIteration], None] | None = None,
    ) -> RetrievalLoopResult:
        run_start = timer_start()
        bounds = self._loop_bounds(max_iterations, max_candidates_per_iteration)

        self._log_start(
            task,
            previous_result,
            bounds.iteration_count,
            bounds.candidate_limit,
        )
        timings_ms = dict(previous_result.timings_ms) if previous_result else {}
        previous_total_ms = timings_ms.pop("total", 0.0)
        description_hints = _validated_description_hints(
            task,
            task_description_hints,
        )
        hint_started_at = (
            self.description_hint_time_source() if description_hints else 0.0
        )
        hint_elapsed_offset_seconds = previous_total_ms / 1000
        initial_hint_index = (
            _description_hint_index(hint_elapsed_offset_seconds)
            if description_hints
            else None
        )
        internal_task = _task_without_answer_text(task)
        if initial_hint_index is not None:
            internal_task = _task_with_description(
                internal_task,
                description_hints[initial_hint_index],
            )

        encoded_step = await self._encode_task(
            internal_task,
            encoded_task,
            previous_result,
        )
        add_timing(timings_ms, "task_encoder", encoded_step.duration_ms)
        self._log_encoder_done(encoded_step)

        full_history = self._initial_history(
            internal_task,
            encoded_step,
            previous_result,
        )

        def total_elapsed_ms() -> float:
            return round(previous_total_ms + elapsed_ms(run_start), 3)

        if self._encoding_is_pending(encoded_step.encoded, internal_task):
            timings_ms["total"] = total_elapsed_ms()
            return self._encode_pending_result(
                internal_task,
                encoded_step.encoded,
                full_history,
                timings_ms,
            )

        run = self._run_context(
            task=task,
            internal_task=internal_task,
            encoded=encoded_step.encoded,
            timings_ms=timings_ms,
            full_history=full_history,
            previous_result=previous_result,
            description_hints=description_hints,
            description_hint_started_at=hint_started_at,
            description_hint_elapsed_offset_seconds=hint_elapsed_offset_seconds,
            description_hint_index=initial_hint_index,
            on_iteration_progress=on_iteration_progress,
        )
        outcome = await self._run_until_found(
            run,
            task_type=task.task_type,
            bounds=bounds,
            on_iteration_start=on_iteration_start,
            on_iteration_complete=on_iteration_complete,
        )
        if outcome is not None:
            timings_ms["total"] = total_elapsed_ms()
            self._log_found(outcome, timings_ms["total"])
            return self._found_result(task.task_type, run, outcome)

        timings_ms["total"] = total_elapsed_ms()
        return self._exhausted_result(task.task_type, run, timings_ms)

    def _encode_pending_result(
        self,
        task: TaskCreate,
        encoded: EncodedTask,
        full_history: list[RetrievalHistoryEntry],
        timings_ms: dict[str, float],
    ) -> RetrievalLoopResult:
        logger.info(
            "retrieval_loop.encode_pending status=%s has_description=%s total_ms=%.3f",
            encoded.status,
            bool(_task_description(encoded, task)),
            timings_ms["total"],
        )
        return RetrievalLoopResult(
            status=RetrievalLoopStatus.ENCODE_PENDING,
            encoded_task=encoded,
            full_history=full_history,
            timings_ms=timings_ms,
        )

    def _exhausted_result(
        self,
        task_type: TaskType,
        run: RunContext,
        timings_ms: dict[str, float],
    ) -> RetrievalLoopResult:
        final_status = (
            RetrievalLoopStatus.FOUND
            if task_type == TaskType.AVS and run.avs_matches_by_id
            else RetrievalLoopStatus.NOT_FOUND
        )
        logger.info(
            "retrieval_loop.done status=%s iterations=%s avs_matches=%s total_ms=%.3f",
            final_status,
            len(run.iterations),
            len(run.avs_matches_by_id),
            timings_ms["total"],
        )
        return RetrievalLoopResult(
            status=final_status,
            encoded_task=run.encoded,
            iterations=run.iterations,
            full_history=run.full_history,
            condensed_planner_history=run.condensed_history,
            avs_matches=list(run.avs_matches_by_id.values()),
            timings_ms=timings_ms,
        )

    def _loop_bounds(
        self,
        max_iterations: int | None,
        max_candidates_per_iteration: int | None,
    ) -> _LoopBounds:
        iteration_count = (
            self.max_iterations if max_iterations is None else max_iterations
        )
        candidate_limit = (
            self.max_candidates_per_iteration
            if max_candidates_per_iteration is None
            else max_candidates_per_iteration
        )
        _validate_limit("max_iterations", iteration_count)
        _validate_limit("max_candidates_per_iteration", candidate_limit)
        return _LoopBounds(
            iteration_count=iteration_count,
            candidate_limit=candidate_limit,
        )

    def _run_context(
        self,
        *,
        task: TaskCreate,
        internal_task: TaskCreate,
        encoded: EncodedTask,
        timings_ms: dict[str, float],
        full_history: list[RetrievalHistoryEntry],
        previous_result: RetrievalLoopResult | None,
        description_hints: tuple[str, ...],
        description_hint_started_at: float,
        description_hint_elapsed_offset_seconds: float,
        description_hint_index: int | None,
        on_iteration_progress: Callable[[RetrievalLoopIteration], None] | None,
    ) -> RunContext:
        run = RunContext(
            task=internal_task,
            validation_task=task,
            encoded=encoded,
            timings_ms=timings_ms,
            iterations=(list(previous_result.iterations) if previous_result else []),
            full_history=full_history,
            condensed_history=(
                previous_result.condensed_planner_history.model_copy(deep=True)
                if previous_result
                else PlannerCondensedHistory()
            ),
            description_hints=description_hints,
            description_hint_started_at=(
                description_hint_started_at if description_hints else None
            ),
            description_hint_elapsed_offset_seconds=(
                description_hint_elapsed_offset_seconds if description_hints else 0.0
            ),
            description_hint_index=description_hint_index,
            irrelevant_candidate_ids=self._previous_irrelevant_candidate_ids(
                previous_result
            ),
            planner_reference_candidates=(
                self._previous_planner_reference_candidates(previous_result)
                if self.planner.keyframe_references_enabled
                else []
            ),
            avs_matches_by_id=self._previous_avs_matches(previous_result),
            seen_candidate_ids=self._previous_seen_candidate_ids(previous_result),
            vqa_searched_video_ids=(
                _vqa_searched_video_ids(previous_result) if previous_result else set()
            ),
            vqa_answer_feedback=(
                _previous_vqa_answer_feedback(previous_result)
                if previous_result
                else []
            ),
            rejected_vqa_answers=self._previous_rejected_vqa_answers(
                task,
                previous_result,
            ),
            submitted_vqa_answers=(
                _submitted_vqa_answers(previous_result) if previous_result else set()
            ),
            submitted_candidates=(
                self._previous_submitted_candidates(previous_result)
                if previous_result and task.task_type == TaskType.AVS
                else []
            ),
            on_iteration_progress=on_iteration_progress,
        )
        for answer in run.rejected_vqa_answers:
            _mark_vqa_answer_rejected(run, answer)
            _record_vqa_answer_feedback(
                run,
                _VQAAnswerFeedback(answer=answer, status="rejected"),
            )
        return run

    def _previous_avs_matches(
        self,
        previous_result: RetrievalLoopResult | None,
    ) -> dict[str, AVSMatch]:
        if previous_result is None:
            return {}
        return {
            candidate_id(match.candidate): match
            for match in previous_result.avs_matches
        }

    def _previous_seen_candidate_ids(
        self,
        previous_result: RetrievalLoopResult | None,
    ) -> set[str]:
        if previous_result is None:
            return set()
        return {
            candidate_id(candidate)
            for iteration in previous_result.iterations
            for candidate in iteration.candidates
        }

    def _previous_irrelevant_candidate_ids(
        self,
        previous_result: RetrievalLoopResult | None,
    ) -> set[str]:
        if previous_result is None:
            return set()
        return {
            result.candidate_id
            for iteration in previous_result.iterations
            if iteration.browser_output is not None
            for result in iteration.browser_output.results
            if result.judgement == "irrelevant"
        }

    def _previous_submitted_candidates(
        self,
        previous_result: RetrievalLoopResult | None,
    ) -> list[CandidateSubmission]:
        if previous_result is None:
            return []
        return [
            submission
            for iteration in previous_result.iterations
            for submission in iteration.submissions
        ]

    def _previous_planner_reference_candidates(
        self,
        previous_result: RetrievalLoopResult | None,
    ) -> list[Candidate]:
        """Restore references from the latest prior retrieval that returned frames."""

        if previous_result is None:
            return []
        evidence_ids = {
            evidence.keyframe_id
            for evidence in previous_result.condensed_planner_history.keyframe_evidence
        }
        for iteration in reversed(previous_result.iterations):
            if not _iteration_renewed_planner_references(iteration):
                continue
            return _rated_reference_candidates(iteration, evidence_ids)
        return []

    def _previous_rejected_vqa_answers(
        self,
        task: TaskCreate,
        previous_result: RetrievalLoopResult | None,
    ) -> list[str]:
        if (
            previous_result is None
            or task.task_type != TaskType.QA
            or not previous_result.final_answer
        ):
            return []
        return [previous_result.final_answer]

    async def _run_until_found(
        self,
        run: RunContext,
        *,
        task_type: TaskType,
        bounds: _LoopBounds,
        on_iteration_start: Callable[[int], None] | None,
        on_iteration_complete: Callable[[RetrievalLoopIteration], None] | None,
    ) -> IterationOutcome | None:
        start_index = len(run.iterations) + 1
        end_index = start_index + bounds.iteration_count
        for index in range(start_index, end_index):
            hint_changed = await self._advance_description_hint(run, iteration=index)
            logger.info("retrieval_loop.iteration iteration=%s", index)
            if on_iteration_start is not None:
                on_iteration_start(index)
            if hint_changed and run.deferred_match_candidates:
                outcome = await self._reevaluate_deferred_matches(run, index=index)
            else:
                outcome = await self._run_iteration(
                    run,
                    index=index,
                    candidate_limit=bounds.candidate_limit,
                )
            await self._record_iteration_outcome(run, outcome, index, task_type)
            if on_iteration_complete is not None:
                on_iteration_complete(outcome.iteration)
            if outcome.found:
                return outcome
        return None

    async def _advance_description_hint(
        self,
        run: RunContext,
        *,
        iteration: int,
    ) -> bool:
        if not run.description_hints or run.description_hint_started_at is None:
            return False

        elapsed_seconds = run.description_hint_elapsed_offset_seconds + max(
            0.0,
            self.description_hint_time_source() - run.description_hint_started_at,
        )
        hint_index = _description_hint_index(elapsed_seconds)
        if hint_index == run.description_hint_index:
            return False

        previous_hint_index = run.description_hint_index
        task = _task_with_description(run.task, run.description_hints[hint_index])
        encoder_start = timer_start()
        encoded = await self.encoder.encode(task)
        encoder_duration = elapsed_ms(encoder_start)
        if self._encoding_is_pending(encoded, task):
            raise ValueError(
                f"Could not encode iterative T-KIS hint {hint_index} for task "
                f"{task.source_task_name!r}."
            )

        add_timing(run.timings_ms, "task_encoder", encoder_duration)
        run.task = task
        run.encoded = encoded
        run.description_hint_index = hint_index
        cleared_history_steps = len(run.condensed_history)
        run.condensed_history = PlannerCondensedHistory()
        run.planner_reference_candidates = []
        run.seen_candidate_ids = set()
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="task_hint",
                kind="module",
                request={"hint_index": previous_hint_index},
                response={
                    "hint_index": hint_index,
                    "description": task.input.description,
                    "cleared_planner_history_steps": cleared_history_steps,
                    "irrelevant_candidate_count": len(run.irrelevant_candidate_ids),
                    "deferred_match_count": len(run.deferred_match_candidates),
                },
                metadata={
                    "duration_ms": encoder_duration,
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "iteration": iteration,
                },
            )
        )
        logger.info(
            "retrieval_loop.task_hint iteration=%s hint_index=%s elapsed_seconds=%.3f",
            iteration,
            hint_index,
            elapsed_seconds,
        )
        return True

    async def _reevaluate_deferred_matches(
        self,
        run: RunContext,
        *,
        index: int,
    ) -> IterationOutcome:
        iteration_start = timer_start()
        candidates = list(run.deferred_match_candidates.values())
        run.deferred_match_candidates.clear()
        action = (
            f"Re-evaluate deferred matches for T-KIS hint {run.description_hint_index}"
        )
        context = IterationContext(
            run=run,
            decision=PlannerDecision(
                action=action,
                purpose="Re-evaluate matches retained from the previous hint phase.",
                policy_rule="hint_transition",
                query=TextQuery(
                    dataset=run.task.dataset,
                    query=run.task.input.description or "",
                    candidateCount=max(1, len(candidates)),
                ),
                endpoint="hintReevaluation",
            ),
            iteration=RetrievalLoopIteration(
                index=index,
                action=action,
                endpoint="hintReevaluation",
                candidates=candidates,
            ),
            started_at=iteration_start,
        )
        inspection = await self.browser_pipeline.inspect(context, candidates)
        self._remember_irrelevant_candidates(run, inspection)
        return await self._handle_deferred_match_inspection(context, inspection)

    async def _record_iteration_outcome(
        self,
        run: RunContext,
        outcome: IterationOutcome,
        index: int,
        task_type: TaskType,
    ) -> None:
        if (
            self.planner.keyframe_references_enabled
            and _iteration_renewed_planner_references(outcome.iteration)
        ):
            evidence_ids = {
                evidence.keyframe_id
                for evidence in run.condensed_history.keyframe_evidence
            }
            run.planner_reference_candidates = _rated_reference_candidates(
                outcome.iteration,
                evidence_ids,
            )
        run.iterations.append(outcome.iteration)
        run.seen_candidate_ids.update(
            candidate_id(candidate) for candidate in outcome.iteration.candidates
        )
        for match in outcome.avs_matches:
            run.avs_matches_by_id.setdefault(candidate_id(match.candidate), match)
        if task_type == TaskType.AVS:
            run.submitted_candidates.extend(outcome.iteration.submissions)
        if task_type not in {TaskType.AVS, TaskType.QA}:
            await self.answerer_observer.run(run, outcome.iteration)
        logger.info(
            "retrieval_loop.answer iteration=%s answer_found=%s",
            index,
            str(outcome.found).lower(),
        )

    def _found_result(
        self,
        task_type: TaskType,
        run: RunContext,
        outcome: IterationOutcome,
    ) -> RetrievalLoopResult:
        return RetrievalLoopResult(
            status=RetrievalLoopStatus.FOUND,
            encoded_task=run.encoded,
            iterations=run.iterations,
            full_history=run.full_history,
            condensed_planner_history=run.condensed_history,
            final_candidate=(
                None if task_type == TaskType.QA else outcome.final_candidate
            ),
            final_answer=outcome.final_answer,
            final_evaluation=outcome.final_evaluation,
            final_submission=(
                None if task_type == TaskType.QA else outcome.final_submission
            ),
            avs_matches=list(run.avs_matches_by_id.values()),
            timings_ms=run.timings_ms,
        )

    async def _run_iteration(
        self,
        run: RunContext,
        *,
        index: int,
        candidate_limit: int,
    ) -> IterationOutcome:
        iteration_start = timer_start()
        logger.info(
            "retrieval_loop.iteration.start iteration=%s history_steps=%s",
            index,
            len(run.condensed_history),
        )

        planned = await self.search_stages.plan(run, index, iteration_start)
        if isinstance(planned, RetrievalLoopIteration):
            return IterationOutcome(iteration=planned)
        context = planned

        action_outcome = await self._run_planner_action(context)
        if action_outcome is not None:
            return action_outcome

        candidates = await self._retrieve_candidates(context, candidate_limit)
        if not candidates:
            return IterationOutcome(iteration=context.iteration)

        inspection = await self.browser_pipeline.inspect(context, candidates)
        self._remember_irrelevant_candidates(run, inspection)
        if run.task.task_type == TaskType.AVS:
            return await self._handle_avs_inspection(context, inspection)
        if run.task.task_type == TaskType.QA and inspection.exact_candidates:
            return await self._handle_qa_inspection(context, inspection)
        return await self._handle_standard_inspection(context, inspection)

    async def _run_planner_action(
        self,
        context: IterationContext,
    ) -> IterationOutcome | None:
        run = context.run
        if isinstance(context.decision.query, DirectAnswer):
            if run.task.task_type == TaskType.AVS:
                return await self.action_handlers.run_avs_direct_answer(context)
            if run.task.task_type == TaskType.QA:
                return await self._answer_qa_direct_answer(context)
            return await self.action_handlers.run_direct_answer(context)
        if isinstance(context.decision.query, (ImageQAAction, VideoQAAction)):
            return await self.action_handlers.run_qa(context)
        return None

    async def _retrieve_candidates(
        self,
        context: IterationContext,
        candidate_limit: int,
    ) -> list[Candidate]:
        run = context.run
        retrieved = await self.search_stages.retrieve(
            context,
            candidate_limit=candidate_limit,
        )
        if retrieved is None:
            return []

        context.iteration.retrieved_candidate_ids = candidate_ids(retrieved.candidates)
        candidates = [
            candidate
            for candidate in unique_candidates(retrieved.candidates)
            if candidate_id(candidate) not in run.irrelevant_candidate_ids
        ]
        context.iteration.filtered_candidates = [
            candidate
            for candidate in candidates
            if candidate_id(candidate) in run.seen_candidate_ids
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate_id(candidate) not in run.seen_candidate_ids
        ]
        candidates = candidates[:candidate_limit]
        context.iteration.candidates = candidates
        self.search_stages.record_success(context, retrieved, candidates)
        if not candidates:
            self.search_stages.finish_empty(context)
        return candidates

    async def _handle_avs_inspection(
        self,
        context: IterationContext,
        inspection: BrowserInspection,
    ) -> IterationOutcome:
        avs_matches: list[AVSMatch] = []
        for candidate in inspection.exact_candidates:
            if _candidate_scene_was_submitted(context, candidate):
                continue
            evaluation = _browser_exact_evaluation(inspection, candidate)
            context.iteration.evaluator_results.append(evaluation)
            submission = self.verification_pipeline.verify(context, candidate)
            if submission.accepted:
                avs_matches.append(AVSMatch(candidate=candidate, evaluation=evaluation))
        await self.browser_pipeline.record_feedback(
            context,
            inspection,
            include_accepted_evaluations=True,
        )
        context.finish()
        logger.info(
            "retrieval_loop.avs.iteration.done iteration=%s total_ms=%.3f "
            "accepted=%s evaluated=%s",
            context.index,
            context.iteration.timings_ms["total"],
            len(avs_matches),
            len(context.iteration.evaluator_results),
        )
        return IterationOutcome(
            iteration=context.iteration,
            avs_matches=tuple(avs_matches),
        )

    async def _handle_qa_inspection(
        self,
        context: IterationContext,
        inspection: BrowserInspection,
    ) -> IterationOutcome:
        last_outcome: IterationOutcome | None = None
        for candidate in inspection.exact_candidates:
            evaluation = _browser_exact_evaluation(inspection, candidate)
            context.iteration.evaluator_results.append(evaluation)
            last_outcome = await self._answer_qa_candidate(
                context=context,
                inspection=inspection,
                accepted_evaluation=evaluation,
                final_candidate=candidate,
            )
            if last_outcome.found:
                return last_outcome
        if last_outcome is None:
            await self.browser_pipeline.record_feedback(
                context,
                inspection,
                include_accepted_evaluations=True,
            )
        context.finish()
        return IterationOutcome(iteration=context.iteration)

    async def _handle_standard_inspection(
        self,
        context: IterationContext,
        inspection: BrowserInspection,
    ) -> IterationOutcome:
        if context.run.description_hints:
            return await self._handle_iterative_hint_inspection(context, inspection)

        for candidate in inspection.exact_candidates:
            evaluation = _browser_exact_evaluation(inspection, candidate)
            context.iteration.evaluator_results.append(evaluation)
            submission = self.verification_pipeline.verify(context, candidate)
            if submission.accepted:
                context.finish()
                return IterationOutcome(
                    iteration=context.iteration,
                    final_candidate=candidate,
                    final_evaluation=evaluation,
                    final_submission=submission,
                )

        await self.browser_pipeline.record_feedback(context, inspection)
        context.finish()
        logger.info(
            "retrieval_loop.iteration.done iteration=%s total_ms=%.3f accepted=false",
            context.index,
            context.iteration.timings_ms["total"],
        )
        return IterationOutcome(iteration=context.iteration)

    async def _handle_iterative_hint_inspection(
        self,
        context: IterationContext,
        inspection: BrowserInspection,
    ) -> IterationOutcome:
        exact_candidates = list(inspection.exact_candidates)
        for candidate in exact_candidates:
            context.iteration.evaluator_results.append(
                _browser_exact_evaluation(inspection, candidate)
            )

        eligible_candidates = [
            candidate
            for candidate in exact_candidates
            if not _candidate_scene_was_submitted(context, candidate)
        ]
        if _uses_deferred_match_submission(context.run):
            matching_video_ids = {
                video_id
                for candidate in eligible_candidates
                if (video_id := _candidate_video_id(candidate)) is not None
            }
            if len(matching_video_ids) > 1:
                context.run.deferred_match_candidates.update(
                    (candidate_id(candidate), candidate)
                    for candidate in eligible_candidates
                )
                candidates_to_submit: list[Candidate] = []
            elif eligible_candidates:
                candidates_to_submit = [_highest_ranked_candidate(eligible_candidates)]
            else:
                candidates_to_submit = []
        else:
            candidates_to_submit = eligible_candidates

        accepted_result: (
            tuple[
                Candidate,
                CandidateEvaluationResult,
                CandidateSubmission,
            ]
            | None
        ) = None
        evaluations = {
            result.candidate_id: result
            for result in context.iteration.evaluator_results
        }
        for candidate in candidates_to_submit:
            submission = self.verification_pipeline.verify(context, candidate)
            if submission.accepted and accepted_result is None:
                accepted_result = (
                    candidate,
                    evaluations[candidate_id(candidate)],
                    submission,
                )

        if accepted_result is not None:
            selected, evaluation, submission = accepted_result
            context.finish()
            return IterationOutcome(
                iteration=context.iteration,
                final_candidate=selected,
                final_evaluation=evaluation,
                final_submission=submission,
            )

        await self.browser_pipeline.record_feedback(context, inspection)
        context.finish()
        logger.info(
            "retrieval_loop.iteration.done iteration=%s total_ms=%.3f "
            "accepted=false deferred_matches=%s",
            context.index,
            context.iteration.timings_ms["total"],
            len(context.run.deferred_match_candidates),
        )
        return IterationOutcome(iteration=context.iteration)

    async def _handle_deferred_match_inspection(
        self,
        context: IterationContext,
        inspection: BrowserInspection,
    ) -> IterationOutcome:
        exact_candidates = list(inspection.exact_candidates)
        evaluations = {
            candidate_id(candidate): _browser_exact_evaluation(inspection, candidate)
            for candidate in exact_candidates
        }
        context.iteration.evaluator_results.extend(evaluations.values())

        if _uses_deferred_match_submission(context.run):
            matching_video_ids = {
                video_id
                for candidate in exact_candidates
                if (video_id := _candidate_video_id(candidate)) is not None
            }
            if len(matching_video_ids) > 1:
                context.run.deferred_match_candidates.update(
                    (candidate_id(candidate), candidate)
                    for candidate in exact_candidates
                )
                candidates_to_submit: list[Candidate] = []
            elif exact_candidates:
                candidates_to_submit = [_highest_ranked_candidate(exact_candidates)]
            else:
                candidates_to_submit = []
        else:
            candidates_to_submit = exact_candidates

        accepted_result: tuple[Candidate, CandidateSubmission] | None = None
        for candidate in candidates_to_submit:
            if _candidate_scene_was_submitted(context, candidate):
                continue
            submission = self.verification_pipeline.verify(context, candidate)
            if submission.accepted and accepted_result is None:
                accepted_result = candidate, submission

        if accepted_result is not None:
            candidate, submission = accepted_result
            context.finish()
            return IterationOutcome(
                iteration=context.iteration,
                final_candidate=candidate,
                final_evaluation=evaluations[candidate_id(candidate)],
                final_submission=submission,
            )

        context.finish()
        logger.info(
            "retrieval_loop.hint_reevaluation.done iteration=%s total_ms=%.3f "
            "accepted=false retained_matches=%s",
            context.index,
            context.iteration.timings_ms["total"],
            len(context.run.deferred_match_candidates),
        )
        return IterationOutcome(iteration=context.iteration)

    @staticmethod
    def _remember_irrelevant_candidates(
        run: RunContext,
        inspection: BrowserInspection,
    ) -> None:
        run.irrelevant_candidate_ids.update(inspection.output.irrelevant_candidate_ids)

    async def _answer_qa_direct_answer(
        self,
        context: IterationContext,
    ) -> IterationOutcome:
        query = context.decision.query
        if not isinstance(query, DirectAnswer):
            raise TypeError("QA direct-answer handling requires a DirectAnswer query.")

        candidate = await self.action_handlers._materialize_direct_answer(
            context,
            query,
        )
        evaluation = CandidateEvaluationResult(
            candidate_id=candidate_id(candidate),
            matches_task=True,
            reasoning=(
                "Planner selected this known video/timestamp for QA final VQA; no "
                "image or video submission was performed."
            ),
        )
        context.iteration.evaluator_results.append(evaluation)
        return await self._answer_qa_candidate(
            context=context,
            inspection=None,
            accepted_evaluation=evaluation,
            final_candidate=candidate,
        )

    async def _answer_qa_candidate(
        self,
        *,
        context: IterationContext,
        inspection: BrowserInspection | None,
        accepted_evaluation: CandidateEvaluationResult,
        final_candidate: Candidate,
    ) -> IterationOutcome:
        source_video_id = _candidate_video_id(final_candidate)
        force_temporal_mode = (
            "V"
            if source_video_id and source_video_id in context.run.vqa_searched_video_ids
            else None
        )

        if self.vqa_planner_session is not None:
            return await self._run_vqa_planner_session(
                context=context,
                inspection=inspection,
                evaluation=accepted_evaluation,
                matched_candidate=final_candidate,
                source_video_id=source_video_id,
            )

        attempt_state = await self._run_vqa_attempts(
            context,
            accepted_evaluation,
            final_candidate,
            source_video_id,
            force_temporal_mode=force_temporal_mode,
        )
        accepted_evaluation.metadata["vqa_attempts"] = [
            result.as_dict() for result in attempt_state.results
        ]
        final_vqa_result = attempt_state.results[-1]
        accepted_evaluation.metadata["vqa"] = final_vqa_result.as_dict()

        if attempt_state.accepted_result is None:
            return self._finalize_rejected_vqa_candidate(
                context,
                inspection,
                accepted_evaluation,
                attempt_state,
                final_vqa_result,
                source_video_id,
            )
        return self._finalize_accepted_vqa_candidate(
            context,
            accepted_evaluation,
            attempt_state.accepted_result,
        )

    async def _run_vqa_planner_session(
        self,
        *,
        context: IterationContext,
        inspection: BrowserInspection | None,
        evaluation: CandidateEvaluationResult,
        matched_candidate: Candidate,
        source_video_id: str | None,
    ) -> IterationOutcome:
        session = self.vqa_planner_session
        if session is None:
            raise RuntimeError("VQA planner session is not configured.")

        run = context.run

        def validate_answer(answer: str):
            validation_start = timer_start()
            validation = verify_text_answer(
                task=run.validation_task,
                answer=answer,
            )
            duration = elapsed_ms(validation_start)
            add_timing(run.timings_ms, "vqa_planner.answer_submission", duration)
            add_timing(
                context.iteration.timings_ms,
                "vqa_planner.answer_submission",
                duration,
            )
            run.submitted_vqa_answers.add(_normalized_answer(answer))
            if validation.status != AnswererValidationStatus.ACCEPTED:
                _mark_vqa_answer_rejected(run, answer)
            return validation

        planner_started = timer_start()
        matched_scene_description = _vqa_matched_scene_description(
            context,
            inspection,
            matched_candidate,
        )
        result = await session.run(
            task=run.task,
            encoded_task=run.encoded,
            matched_candidate=matched_candidate,
            question=_task_question(run.task),
            validate_answer=validate_answer,
            matched_scene_description=matched_scene_description,
        )
        duration = elapsed_ms(planner_started)
        add_timing(run.timings_ms, "vqa_planner", duration)
        add_timing(context.iteration.timings_ms, "vqa_planner", duration)
        evaluation.metadata["vqa_planner"] = result.model_dump(mode="json")
        self._record_vqa_planner_history(context, result)

        if result.answer is not None:
            evaluation.metadata["answer"] = result.answer
            evaluation.metadata["answer_source"] = "vqa_planner"
            evaluation.reasoning = result.answer
            context.finish()
            return IterationOutcome(
                iteration=context.iteration,
                final_answer=result.answer,
                final_evaluation=evaluation,
                final_submission=None,
            )

        reason = _qa_video_not_found_reason(source_video_id, result.reason)
        self._reject_qa_vqa_candidate(
            context=context,
            inspection=inspection,
            evaluation=evaluation,
            source_video_id=source_video_id,
            reason=reason,
        )
        context.finish()
        return IterationOutcome(iteration=context.iteration)

    def _record_vqa_planner_history(
        self,
        context: IterationContext,
        result: VQAPlannerSessionResult,
    ) -> None:
        for step in result.steps:
            context.run.full_history.extend(
                llm_history_entries(
                    "vqa_planner",
                    step.decision.metadata,
                    iteration=context.index,
                )
            )
            context.run.full_history.extend(
                llm_history_entries(
                    "vqa_planner.operation",
                    step.metadata,
                    iteration=context.index,
                )
            )
            context.run.full_history.append(
                RetrievalHistoryEntry(
                    stage=f"vqa_planner.{step.decision.action.action}",
                    kind="module",
                    request=step.decision.action.model_dump(mode="json"),
                    response=step.result,
                    metadata={
                        "iteration": context.index,
                        "step": step.index,
                        "duration_ms": step.metadata.get("duration_ms"),
                    },
                )
            )
        context.run.full_history.append(
            RetrievalHistoryEntry(
                stage="vqa_planner.session",
                kind="module",
                request={
                    "question": _task_question(context.run.task),
                    "invocation_timestamp_ms": result.metadata.get(
                        "invocation_timestamp_ms"
                    ),
                    "matched_scene_description": result.metadata.get(
                        "matched_scene_description"
                    ),
                },
                response={
                    "answer": result.answer,
                    "ended_without_answer": result.ended_without_answer,
                    "reason": result.reason,
                    "step_count": len(result.steps),
                },
                metadata={
                    "iteration": context.index,
                    **result.metadata,
                },
            )
        )

    async def _run_vqa_attempts(
        self,
        context: IterationContext,
        evaluation: CandidateEvaluationResult,
        final_candidate: Candidate,
        source_video_id: str | None,
        *,
        force_temporal_mode: str | None = None,
    ) -> _VQAAttemptState:
        run = context.run
        attempt_state = _VQAAttemptState(
            results=[],
            answer_feedback=_vqa_answer_feedback_from_run(run),
            rejected_answer_reasons=[],
        )
        attempt_index = 1
        while (
            attempt_index <= self.max_vqa_attempts
            or attempt_state.nonaccepted_retry_pending
        ):
            should_stop = await self._run_vqa_attempt(
                context=context,
                evaluation=evaluation,
                final_candidate=final_candidate,
                source_video_id=source_video_id,
                attempt_index=attempt_index,
                attempt_state=attempt_state,
                force_temporal_mode=force_temporal_mode,
            )
            if should_stop:
                break
            attempt_index += 1
        return attempt_state

    def _finalize_rejected_vqa_candidate(
        self,
        context: IterationContext,
        inspection: BrowserInspection | None,
        evaluation: CandidateEvaluationResult,
        attempt_state: _VQAAttemptState,
        final_vqa_result: VQAResult,
        source_video_id: str | None,
    ) -> IterationOutcome:
        reason = _qa_video_not_found_reason(
            source_video_id,
            (
                "The generated text answers were not accepted by the outside "
                "answer checker."
                if attempt_state.rejected_answer_reasons
                else final_vqa_result.reason
            ),
        )
        self._reject_qa_vqa_candidate(
            context=context,
            inspection=inspection,
            evaluation=evaluation,
            source_video_id=source_video_id,
            reason=reason,
        )
        context.finish()
        return IterationOutcome(iteration=context.iteration)

    def _finalize_accepted_vqa_candidate(
        self,
        context: IterationContext,
        evaluation: CandidateEvaluationResult,
        vqa_result: VQAResult,
    ) -> IterationOutcome:
        evaluation.metadata["answer"] = vqa_result.answer
        evaluation.reasoning = vqa_result.answer
        context.finish()
        return IterationOutcome(
            iteration=context.iteration,
            final_answer=vqa_result.answer,
            final_evaluation=evaluation,
            final_submission=None,
        )

    async def _run_vqa_attempt(
        self,
        *,
        context: IterationContext,
        evaluation: CandidateEvaluationResult,
        final_candidate: Candidate,
        source_video_id: str | None,
        attempt_index: int,
        attempt_state: _VQAAttemptState,
        force_temporal_mode: str | None = None,
    ) -> bool:
        run = context.run
        vqa_start = timer_start()
        try:
            def record_progress(vqa_result: VQAResult) -> None:
                partial_attempts = [
                    result.as_dict() for result in attempt_state.results
                ]
                partial_attempts.append(vqa_result.as_dict())
                evaluation.metadata["vqa_attempts"] = partial_attempts
                evaluation.metadata["vqa"] = vqa_result.as_dict()
                context.iteration.timings_ms["total"] = elapsed_ms(context.started_at)
                if run.on_iteration_progress is not None:
                    try:
                        run.on_iteration_progress(context.iteration)
                    except Exception:
                        logger.exception(
                            "retrieval_loop.iteration_progress.failed iteration=%s",
                            context.index,
                        )

            vqa_result = await self._run_final_vqa(
                run,
                final_candidate,
                attempt_state.answer_feedback,
                previous_vqa_results=attempt_state.results,
                force_temporal_mode=force_temporal_mode,
                progress_callback=record_progress,
            )
        except Exception:
            logger.exception(
                "retrieval_loop.vqa.failed iteration=%s attempt=%s duration_ms=%.3f",
                context.index,
                attempt_index,
                elapsed_ms(vqa_start),
            )
            raise
        vqa_duration = elapsed_ms(vqa_start)
        add_timing(run.timings_ms, "vqa", vqa_duration)
        context.iteration.timings_ms["vqa"] = round(
            context.iteration.timings_ms.get("vqa", 0.0) + vqa_duration,
            3,
        )
        attempt_state.results.append(vqa_result)
        self._record_vqa_attempt(
            context=context,
            evaluation=evaluation,
            vqa_result=vqa_result,
            source_video_id=source_video_id,
            attempt_index=attempt_index,
            duration_ms=vqa_duration,
            answer_feedback=tuple(attempt_state.answer_feedback),
        )
        if not vqa_result.answerable or not vqa_result.answer:
            if attempt_state.nonaccepted_retry_pending:
                attempt_state.nonaccepted_retry_pending = False
            return True
        if _vqa_answer_was_submitted(run, vqa_result.answer):
            self._record_duplicate_vqa_answer(
                context,
                source_video_id,
                vqa_result.answer,
                attempt_index,
            )
            attempt_state.nonaccepted_retry_pending = (
                _vqa_n_mode_can_continue(vqa_result)
            )
            return False
        return self._validate_vqa_answer(
            context=context,
            evaluation=evaluation,
            vqa_result=vqa_result,
            attempt_index=attempt_index,
            attempt_state=attempt_state,
            source_video_id=source_video_id,
        )

    def _record_duplicate_vqa_answer(
        self,
        context: IterationContext,
        source_video_id: str | None,
        answer: str,
        attempt_index: int,
    ) -> None:
        context.run.full_history.append(
            RetrievalHistoryEntry(
                stage="vqa.answer_duplicate",
                kind="module",
                request={"answer": answer, "source_video_id": source_video_id},
                response={
                    "submitted": False,
                    "reason": (
                        "This text answer was already submitted earlier in the QA loop."
                    ),
                },
                metadata={"iteration": context.index, "attempt": attempt_index},
            )
        )

    def _validate_vqa_answer(
        self,
        *,
        context: IterationContext,
        evaluation: CandidateEvaluationResult,
        vqa_result: VQAResult,
        attempt_index: int,
        attempt_state: _VQAAttemptState,
        source_video_id: str | None,
    ) -> bool:
        run = context.run
        if vqa_result.answer is None:
            return False
        validation_start = timer_start()
        try:
            answer_validation = verify_text_answer(
                task=run.validation_task,
                answer=vqa_result.answer,
            )
        except Exception:
            logger.exception(
                "retrieval_loop.vqa.answer_submission.failed iteration=%s "
                "attempt=%s duration_ms=%.3f",
                context.index,
                attempt_index,
                elapsed_ms(validation_start),
            )
            raise
        validation_duration = elapsed_ms(validation_start)
        add_timing(run.timings_ms, "vqa.answer_submission", validation_duration)
        add_timing(
            context.iteration.timings_ms,
            "vqa.answer_submission",
            validation_duration,
        )
        run.submitted_vqa_answers.add(_normalized_answer(vqa_result.answer))
        evaluation.metadata["vqa_answer_validation"] = answer_validation.model_dump(
            mode="json"
        )
        answer_accepted = answer_validation.status == AnswererValidationStatus.ACCEPTED
        submission = CandidateSubmission(
            candidate_id=_vqa_answer_submission_id(context.index, attempt_index),
            answer=vqa_result.answer,
            accepted=answer_accepted,
            status=CandidateSubmissionStatus(answer_validation.status.value),
            reason=answer_validation.reason,
            source=answer_validation.source or "text_answer_verifier",
        )
        context.iteration.submissions.append(submission)
        evaluation.metadata.setdefault("vqa_answer_submissions", []).append(
            submission.model_dump(mode="json")
        )
        response = jsonable(answer_validation)
        if isinstance(response, dict):
            response["accepted"] = answer_accepted
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="vqa.answer_submission",
                kind="module",
                request={
                    "answer": vqa_result.answer,
                    "has_answer_text": run.validation_task.input.answer_text
                    is not None,
                },
                response=response,
                metadata={
                    "duration_ms": validation_duration,
                    "iteration": context.index,
                    "source_video_id": source_video_id,
                    "attempt": attempt_index,
                },
            )
        )
        logger.info(
            "retrieval_loop.vqa.answer_submission.done iteration=%s attempt=%s "
            "duration_ms=%.3f accepted=%s",
            context.index,
            attempt_index,
            validation_duration,
            answer_accepted,
        )
        if answer_accepted:
            attempt_state.nonaccepted_retry_pending = False
            attempt_state.accepted_result = vqa_result
            return True

        retried_after_nonaccepted = attempt_state.nonaccepted_retry_pending
        if retried_after_nonaccepted:
            attempt_state.nonaccepted_retry_pending = False

        feedback = _VQAAnswerFeedback(
            answer=vqa_result.answer,
            status=answer_validation.status.value,
            reason=answer_validation.reason,
            **_selected_vqa_segment_feedback(vqa_result),
            scene_notes=_vqa_scene_notes_feedback(vqa_result),
        )
        attempt_state.answer_feedback.append(feedback)
        _record_vqa_answer_feedback(run, feedback)
        attempt_state.rejected_answer_reasons.append(answer_validation.reason)
        if answer_validation.status in {
            AnswererValidationStatus.REJECTED,
            AnswererValidationStatus.UNAVAILABLE,
        }:
            _mark_vqa_answer_rejected(run, vqa_result.answer)
        if answer_validation.status == AnswererValidationStatus.REJECTED:
            evaluation.metadata.setdefault("rejected_vqa_answers", []).append(
                {
                    "answer": vqa_result.answer,
                    "reason": answer_validation.reason,
                    "attempt": attempt_index,
                }
            )
        if answer_validation.status == AnswererValidationStatus.UNAVAILABLE:
            evaluation.metadata.setdefault("unavailable_vqa_answers", []).append(
                {
                    "answer": vqa_result.answer,
                    "reason": answer_validation.reason,
                    "attempt": attempt_index,
                }
            )
        if _vqa_n_mode_can_continue(vqa_result):
            attempt_state.nonaccepted_retry_pending = True
            return False
        if not attempt_state.nonaccepted_retry_used:
            attempt_state.nonaccepted_retry_used = True
            attempt_state.nonaccepted_retry_pending = True
            return False
        return False

    async def _run_final_vqa(
        self,
        run: RunContext,
        final_candidate: Candidate,
        answer_feedback: list[_VQAAnswerFeedback],
        *,
        previous_vqa_results: Sequence[VQAResult] = (),
        force_temporal_mode: str | None = None,
        progress_callback: Callable[[VQAResult], None] | None = None,
    ) -> VQAResult:
        task = _task_with_vqa_answer_feedback(run.task, answer_feedback)
        access = self.task_input_access.vqa
        encoded_task = encoded_task_for_module(run.encoded, task, access)
        if force_temporal_mode:
            encoded_task = encoded_task.model_copy(
                update={
                    "metadata": {
                        **encoded_task.metadata,
                        "qa_temporal_mode": force_temporal_mode,
                        "vqa_refresh_scene_notes": True,
                    }
                }
            )
        kwargs: dict[str, Any] = {
            "encoded_task": encoded_task,
            "task_text": task_description_for_module(run.encoded, task, access) or None,
            "include_task_text": access.has_text_source,
            "task_images": task_images_for_module(run.encoded, access),
            "task_videos": task_videos_for_module(
                task,
                access,
                self.task_reference_video_resolver,
            ),
            "progress_callback": progress_callback,
            "previous_vqa_results": previous_vqa_results,
        }
        parameters = inspect.signature(self.vqa.answer).parameters.values()
        accepts_arbitrary_keywords = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        supported_names = {parameter.name for parameter in parameters}
        supported_kwargs = (
            kwargs
            if accepts_arbitrary_keywords
            else {
                name: value for name, value in kwargs.items() if name in supported_names
            }
        )
        return await self.vqa.answer(task, final_candidate, **supported_kwargs)

    def _record_vqa_attempt(
        self,
        *,
        context: IterationContext,
        evaluation: CandidateEvaluationResult,
        vqa_result: VQAResult,
        source_video_id: str | None,
        attempt_index: int,
        duration_ms: float,
        answer_feedback: tuple[_VQAAnswerFeedback, ...],
    ) -> None:
        run = context.run
        run.full_history.extend(
            llm_history_entries(
                "vqa",
                vqa_result.metadata,
                iteration=context.index,
            )
        )
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="vqa",
                kind="module",
                request={
                    "candidate_id": evaluation.candidate_id,
                    "question": _task_question(run.task),
                    "attempt": attempt_index,
                },
                response={
                    "answerable": vqa_result.answerable,
                    "answer": vqa_result.answer,
                    "reason": vqa_result.reason,
                },
                metadata={
                    "duration_ms": duration_ms,
                    "segment_count": vqa_result.metadata.get("segment_count", 0),
                    "iteration": context.index,
                    "attempt": attempt_index,
                    "max_attempts": self.max_vqa_attempts,
                    "source_video_id": source_video_id,
                    "rejected_answers": [
                        item.answer
                        for item in answer_feedback
                        if item.status in {"rejected", "unavailable"}
                    ],
                    "previous_answer_submissions": [
                        item.as_dict() for item in answer_feedback
                    ],
                },
            )
        )

    def _reject_qa_vqa_candidate(
        self,
        *,
        context: IterationContext,
        inspection: BrowserInspection | None,
        evaluation: CandidateEvaluationResult,
        source_video_id: str | None,
        reason: str,
        record_video_feedback: bool = True,
    ) -> None:
        evaluation.matches_task = False
        evaluation.decision = CandidateEvaluationDecision.REJECT
        evaluation.reasoning = reason
        if source_video_id:
            _mark_vqa_video_searched(context.run, source_video_id)
        if record_video_feedback:
            self._record_vqa_video_not_found(
                context=context,
                evaluation=evaluation,
                source_video_id=source_video_id,
                reason=reason,
            )
        if inspection is None:
            history_step = planner_direct_answer_rejected_history_step(
                action=context.decision.action,
                candidate=context.iteration.candidates[0],
                reason=reason,
                prior_history=context.run.condensed_history,
                purpose=context.decision.purpose,
                policy_rule=context.decision.policy_rule,
            )
        else:
            history_step = planner_feedback_history_step(
                action=context.decision.action,
                purpose=context.decision.purpose,
                policy_rule=context.decision.policy_rule,
                candidates_by_id=inspection.evidence.by_id,
                browser_output=inspection.output,
                browser_reasoning=(
                    context.iteration.browser_reasoning_output
                    or BrowserReasoningOutput()
                ),
                browser_descriptions=inspection.descriptions,
                inspected_video_candidate_ids=set(
                    inspection.inspected_video_candidate_ids
                ),
                inspected_video_intervals_ms={
                    candidate_id: (start_ms, end_ms)
                    for candidate_id, start_ms, end_ms in (
                        inspection.inspected_video_intervals_ms
                    )
                },
                evaluator_results=context.iteration.evaluator_results,
                submissions=[],
                include_accepted_evaluations=True,
            )
        if source_video_id:
            history_step.evaluator_result.evaluator_feedback = (
                f"Final VQA searched video {source_video_id!r} and did not find "
                "an accepted text answer. Do not search this video again for QA."
            )
        context.run.condensed_history.add_step(history_step)

    def _record_vqa_video_not_found(
        self,
        *,
        context: IterationContext,
        evaluation: CandidateEvaluationResult,
        source_video_id: str | None,
        reason: str,
    ) -> None:
        context.run.full_history.append(
            RetrievalHistoryEntry(
                stage="vqa.video_not_found",
                kind="module",
                request={
                    "candidate_id": evaluation.candidate_id,
                    "source_video_id": source_video_id,
                },
                response={
                    "answer_found": False,
                    "reason": reason,
                    "feedback": (
                        "The answer was not found in this video. Do not search this "
                        "video again for final QA."
                    ),
                },
                metadata={"iteration": context.index},
            )
        )

    async def _encode_task(
        self,
        task: TaskCreate,
        encoded_task: EncodedTask | None,
        previous_result: RetrievalLoopResult | None,
    ) -> _EncodedTaskStep:
        encoder_start = timer_start()
        try:
            if previous_result is not None:
                encoded = previous_result.encoded_task
                source = "previous"
            elif encoded_task is not None:
                encoded = encoded_task
                source = "provided"
            else:
                encoded = await self.encoder.encode(task)
                source = "encoder"
        except Exception:
            logger.exception(
                "retrieval_loop.task_encoder.failed dataset=%s task_type=%s "
                "duration_ms=%.3f",
                task.dataset.value,
                task.task_type.value,
                elapsed_ms(encoder_start),
            )
            raise

        duration_ms = (
            0.0 if source in {"previous", "provided"} else elapsed_ms(encoder_start)
        )
        return _EncodedTaskStep(encoded=encoded, duration_ms=duration_ms, source=source)

    def _initial_history(
        self,
        task: TaskCreate,
        encoded_step: _EncodedTaskStep,
        previous_result: RetrievalLoopResult | None,
    ) -> list[RetrievalHistoryEntry]:
        if previous_result is not None:
            history = list(previous_result.full_history)
            history.append(
                RetrievalHistoryEntry(
                    stage="retrieval_loop",
                    kind="module",
                    request={
                        "previous_status": previous_result.status,
                        "previous_iteration_count": len(previous_result.iterations),
                    },
                    response="Continuing retrieval loop from previous history.",
                    metadata={"continued": True},
                )
            )
            return history

        return [
            RetrievalHistoryEntry(
                stage="task_encoder",
                kind="module",
                request=jsonable(task),
                response=jsonable(encoded_step.encoded),
                metadata={
                    "duration_ms": encoded_step.duration_ms,
                    "encoded_task_source": encoded_step.source,
                },
            ),
            *llm_history_entries("task_encoder", encoded_step.encoded.metadata),
        ]

    def _log_start(
        self,
        task: TaskCreate,
        previous_result: RetrievalLoopResult | None,
        max_iterations: int,
        max_candidates_per_iteration: int,
    ) -> None:
        logger.info(
            "retrieval_loop.start dataset=%s task_type=%s previous_result=%s "
            "max_iterations=%s max_candidates_per_iteration=%s",
            task.dataset.value,
            task.task_type.value,
            previous_result is not None,
            max_iterations,
            max_candidates_per_iteration,
        )

    def _log_encoder_done(self, encoded_step: _EncodedTaskStep) -> None:
        logger.info(
            "retrieval_loop.task_encoder.done status=%s duration_ms=%.3f source=%s",
            encoded_step.encoded.status,
            encoded_step.duration_ms,
            encoded_step.source,
        )

    def _log_found(self, outcome: IterationOutcome, total_ms: float) -> None:
        if outcome.final_answer is not None:
            logger.info(
                "retrieval_loop.found iteration=%s final_answer=%r "
                "iteration_total_ms=%.3f total_ms=%.3f",
                outcome.iteration.index,
                outcome.final_answer,
                outcome.iteration.timings_ms["total"],
                total_ms,
            )
            return
        if outcome.final_candidate is None:
            raise ValueError("A found iteration requires a final candidate.")
        candidate_id = candidate_ids([outcome.final_candidate])[0]
        if outcome.final_evaluation is None:
            logger.info(
                "retrieval_loop.found iteration=%s candidate_id=%s "
                "source=direct_answer iteration_total_ms=%.3f total_ms=%.3f",
                outcome.iteration.index,
                candidate_id,
                outcome.iteration.timings_ms["total"],
                total_ms,
            )
        else:
            logger.info(
                "retrieval_loop.found iteration=%s candidate_id=%s "
                "iteration_total_ms=%.3f total_ms=%.3f",
                outcome.iteration.index,
                candidate_id,
                outcome.iteration.timings_ms["total"],
                total_ms,
            )

    @staticmethod
    def _encoding_is_pending(encoded: EncodedTask, task: TaskCreate) -> bool:
        return encoded.status != "encoded" or not _task_description(encoded, task)


def default_retrieval_loop(
    max_iterations: int = 3,
    max_candidates_per_iteration: int = DEFAULT_MAX_CANDIDATES_PER_ITERATION,
    hide_avs_submission_ratings_from_planner: bool = False,
) -> RetrievalLoop:
    """Build the default VLM-backed retrieval loop."""

    return RetrievalLoop(
        encoder=VLMTaskEncoder(),
        planner=configured_prompt_planner(),
        answerer=PromptAnswerer(),
        browser=PromptBrowser(),
        qa=PromptQuestionAnswerer(),
        vqa=PromptVQA(),
        max_iterations=max_iterations,
        max_candidates_per_iteration=max_candidates_per_iteration,
        hide_avs_submission_ratings_from_planner=(
            hide_avs_submission_ratings_from_planner
        ),
    )


def _validate_limit(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")


def _task_description(encoded_task: EncodedTask, task: TaskCreate) -> str:
    return (encoded_task.text or task.input.description or "").strip()


def _task_question(task: TaskCreate) -> str:
    """Extract the final QA question from the existing frontend text field."""

    text = (task.input.description or "").strip()
    marker = "Task question:"
    folded_text = text.casefold()
    folded_marker = marker.casefold()
    marker_index = folded_text.find(folded_marker)
    if marker_index < 0:
        return text

    remainder = text[marker_index + len(marker) :].lstrip()
    next_section = remainder.find("\n\n")
    return remainder[:next_section].strip() if next_section >= 0 else remainder.strip()


def _candidate_video_id(candidate: Candidate) -> str | None:
    if len(candidate.id) > 1:
        return "_".join(candidate.id[:-1])
    if candidate.id:
        return candidate.id[0]
    return None


def _uses_deferred_match_submission(run: RunContext) -> bool:
    """Gate ambiguous submissions while an iterative T-KIS hint is incomplete."""

    return bool(run.description_hints) and (
        run.description_hint_index is not None
        and run.description_hint_index < len(run.description_hints) - 1
    )


def _highest_ranked_candidate(candidates: list[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("Cannot select a highest-ranked candidate from an empty list.")

    def rank_key(indexed_candidate: tuple[int, Candidate]) -> tuple[float, int]:
        index, candidate = indexed_candidate
        try:
            rank = float(candidate.rank)
        except (TypeError, ValueError):
            rank = float("inf")
        return rank, index

    return min(enumerate(candidates), key=rank_key)[1]


def _browser_exact_evaluation(
    inspection: BrowserInspection,
    candidate: Candidate,
) -> CandidateEvaluationResult:
    """Represent an authoritative exact browser verdict in the legacy result field."""

    identifier = candidate_id(candidate)
    classification = next(
        (
            result
            for result in inspection.output.results
            if result.candidate_id == identifier
        ),
        None,
    )
    return CandidateEvaluationResult(
        candidate_id=identifier,
        matches_task=True,
        decision=CandidateEvaluationDecision.ACCEPT,
        reasoning=(
            classification.reasoning
            if classification is not None and classification.reasoning
            else "The authoritative browser scene classifier found an exact match."
        ),
        metadata={
            "source": "browser.scene_analysis",
            "matching_evidence": (
                classification.matching_evidence if classification is not None else []
            ),
            "missing_evidence": (
                classification.missing_evidence if classification is not None else []
            ),
            "contradicting_evidence": (
                classification.contradicting_evidence
                if classification is not None
                else []
            ),
        },
    )


def _candidate_scene_was_submitted(
    context: IterationContext,
    candidate: Candidate,
) -> bool:
    identifier = candidate_id(candidate)
    video_id = _candidate_video_id(candidate)
    if video_id is None:
        return False
    for submission in [
        *context.run.submitted_candidates,
        *(
            submission
            for iteration in context.run.iterations
            for submission in iteration.submissions
        ),
        *context.iteration.submissions,
    ]:
        if submission.candidate_id == identifier:
            return True
        if submission.video_id != video_id:
            continue
        if submission.scene_start_ms is None or submission.scene_end_ms is None:
            if submission.timestamp_ms == candidate.time_keyframe:
                return True
            continue
        if (
            candidate.time_scene_start <= submission.scene_end_ms
            and candidate.time_scene_end >= submission.scene_start_ms
        ):
            return True
    return False


def _iteration_renewed_planner_references(
    iteration: RetrievalLoopIteration,
) -> bool:
    """Whether this iteration fetched at least one ratable keyframe result."""

    returned_candidates = bool(iteration.candidates or iteration.filtered_candidates)
    return returned_candidates and (
        iteration.browser_output is not None or bool(iteration.filtered_candidates)
    )


def _rated_reference_candidates(
    iteration: RetrievalLoopIteration,
    known_evidence_ids: set[str],
) -> list[Candidate]:
    """Keep fetched candidates rated relevant or better, in retrieval order."""

    newly_rated_ids: set[str] = set()
    if iteration.browser_output is not None:
        newly_rated_ids.update(iteration.browser_output.relevant_candidate_ids)
        newly_rated_ids.update(iteration.browser_output.match_candidate_ids)

    eligible_ids = newly_rated_ids | {
        candidate_id(candidate)
        for candidate in iteration.filtered_candidates
        if candidate_id(candidate) in known_evidence_ids
    }
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    for candidate in [*iteration.candidates, *iteration.filtered_candidates]:
        identifier = candidate_id(candidate)
        if identifier not in eligible_ids or identifier in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(identifier)
    return selected


def _validated_description_hints(
    task: TaskCreate,
    hints: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if hints is None:
        return ()
    if task.task_type != TaskType.TEXTUAL_KIS:
        raise ValueError("Iterative task-description hints are only valid for T-KIS.")
    if len(hints) != 3:
        raise ValueError("Iterative T-KIS tasks require exactly three text hints.")

    normalized = tuple(hint.strip() for hint in hints)
    if any(not hint for hint in normalized):
        raise ValueError("Iterative T-KIS text hints must not be empty.")
    return normalized


def _description_hint_index(elapsed_seconds: float) -> int:
    if elapsed_seconds < 60:
        return 0
    if elapsed_seconds < 120:
        return 1
    return 2


def _task_with_description(task: TaskCreate, description: str) -> TaskCreate:
    return task.model_copy(
        update={
            "input": task.input.model_copy(
                update={"description": description},
                deep=True,
            )
        },
        deep=True,
    )


def _task_without_answer_text(task: TaskCreate) -> TaskCreate:
    if task.task_type != TaskType.QA or task.input.answer_text is None:
        return task
    return task.model_copy(
        update={
            "input": task.input.model_copy(update={"answer_text": None}),
        }
    )


def _vqa_matched_scene_description(
    context: IterationContext,
    inspection: BrowserInspection | None,
    candidate: Candidate,
) -> str | None:
    """Return the scene context that caused this candidate to reach final VQA."""

    identifier = candidate_id(candidate)
    if inspection is not None:
        scene_description = inspection.descriptions.description_for(
            identifier,
            "scene",
        )
        if scene_description is not None:
            return scene_description
        frame_description = inspection.descriptions.description_for(
            identifier,
            "frame",
        )
        if frame_description is not None:
            return frame_description

    video_id = _candidate_video_id(candidate)
    if video_id is None:
        return None
    matching_evidence = []
    for evidence in context.run.condensed_history.keyframe_evidence:
        if evidence.video_id != video_id:
            continue
        exact_keyframe = evidence.keyframe_id == identifier
        exact_timestamp = evidence.timestamp_ms == candidate.time_keyframe
        contains_timestamp = (
            evidence.inspected_segment is not None
            and evidence.inspected_segment.start_ms
            <= candidate.time_keyframe
            <= evidence.inspected_segment.end_ms
        )
        if exact_keyframe or exact_timestamp or contains_timestamp:
            matching_evidence.append(
                (
                    0 if exact_keyframe else 1 if exact_timestamp else 2,
                    evidence,
                )
            )
    if not matching_evidence:
        return None
    _, evidence = min(matching_evidence, key=lambda item: item[0])
    return evidence.scene_description or evidence.frame_description


def _mark_vqa_video_searched(run: RunContext, video_id: str) -> None:
    if not video_id:
        return
    run.vqa_searched_video_ids.add(video_id)
    if video_id not in run.condensed_history.vqa_searched_video_ids:
        run.condensed_history.vqa_searched_video_ids.append(video_id)


def _mark_vqa_answer_rejected(run: RunContext, answer: str) -> None:
    normalized = _normalized_answer(answer)
    if not normalized:
        return
    if normalized not in {
        _normalized_answer(item) for item in run.rejected_vqa_answers
    }:
        run.rejected_vqa_answers.append(answer)
    if normalized not in {
        _normalized_answer(item) for item in run.condensed_history.rejected_vqa_answers
    }:
        run.condensed_history.rejected_vqa_answers.append(answer)


def _record_vqa_answer_feedback(
    run: RunContext,
    feedback: _VQAAnswerFeedback,
) -> None:
    normalized = _normalized_answer(feedback.answer)
    if not normalized:
        return
    existing_answers = {
        _normalized_answer(str(item.get("answer", "")))
        for item in run.vqa_answer_feedback
        if isinstance(item, dict)
    }
    if normalized in existing_answers:
        return
    run.vqa_answer_feedback.append(feedback.as_dict())


def _vqa_answer_feedback_from_run(run: RunContext) -> list[_VQAAnswerFeedback]:
    feedback: list[_VQAAnswerFeedback] = []
    seen_answers: set[str] = set()
    for item in run.vqa_answer_feedback:
        parsed = _vqa_answer_feedback_from_dict(item)
        if parsed is None:
            continue
        normalized = _normalized_answer(parsed.answer)
        if not normalized or normalized in seen_answers:
            continue
        feedback.append(parsed)
        seen_answers.add(normalized)
    for answer in run.rejected_vqa_answers:
        normalized = _normalized_answer(answer)
        if not normalized or normalized in seen_answers:
            continue
        feedback.append(_VQAAnswerFeedback(answer=answer, status="rejected"))
        seen_answers.add(normalized)
    return feedback


def _vqa_answer_feedback_from_dict(
    item: dict[str, Any],
) -> _VQAAnswerFeedback | None:
    answer = item.get("answer")
    status = item.get("status")
    if not isinstance(answer, str) or not answer:
        return None
    if not isinstance(status, str) or not status:
        return None
    scene_notes = item.get("scene_notes")
    return _VQAAnswerFeedback(
        answer=answer,
        status=status,
        reason=_string_or_none(item.get("reason")),
        segment_index=_int_or_none(item.get("segment_index")),
        start_seconds=_float_or_none(item.get("start_seconds")),
        end_seconds=_float_or_none(item.get("end_seconds")),
        source_candidate_id=_string_or_none(item.get("source_candidate_id")),
        scene_notes=(
            [note for note in scene_notes if isinstance(note, dict)]
            if isinstance(scene_notes, list)
            else None
        ),
    )


def _vqa_answer_was_submitted(run: RunContext, answer: str) -> bool:
    normalized = _normalized_answer(answer)
    return bool(normalized) and normalized in run.submitted_vqa_answers


def _vqa_n_mode_can_continue(vqa_result: VQAResult) -> bool:
    metadata = vqa_result.metadata
    if metadata.get("qa_temporal_mode") != "N":
        return False
    if metadata.get("n_mode_scene_limit_reached") is True:
        return False
    remaining = metadata.get("n_mode_scene_slots_remaining")
    if isinstance(remaining, int) and not isinstance(remaining, bool):
        return remaining > 0
    seen_count = metadata.get("n_mode_seen_scene_count")
    scene_limit = metadata.get("n_mode_scene_limit")
    if (
        isinstance(seen_count, int)
        and not isinstance(seen_count, bool)
        and isinstance(scene_limit, int)
        and not isinstance(scene_limit, bool)
    ):
        return seen_count < scene_limit
    return False


def _vqa_answer_submission_id(iteration_index: int, attempt_index: int) -> str:
    return f"vqa-answer-{iteration_index}-{attempt_index}"


def _selected_vqa_segment_feedback(
    vqa_result: VQAResult,
) -> dict[str, int | float | str | None]:
    selected_index = vqa_result.metadata.get("selected_segment_index")
    if not isinstance(selected_index, int):
        return {}
    selected_observation = next(
        (
            item
            for item in vqa_result.observations
            if item.index == selected_index
        ),
        None,
    )
    if selected_observation is None:
        return {"segment_index": selected_index}
    return {
        "segment_index": selected_observation.index,
        "start_seconds": selected_observation.start_seconds,
        "end_seconds": selected_observation.end_seconds,
        "source_candidate_id": _selected_vqa_segment_source_candidate_id(
            vqa_result,
            selected_index,
        ),
    }


def _selected_vqa_segment_source_candidate_id(
    vqa_result: VQAResult,
    selected_index: int,
) -> str | None:
    calls = vqa_result.metadata.get("llm_calls")
    if not isinstance(calls, list):
        return None
    for call in calls:
        if not isinstance(call, dict):
            continue
        if call.get("segment_index") != selected_index:
            continue
        source_candidate_id = call.get("source_candidate_id")
        if isinstance(source_candidate_id, str) and source_candidate_id:
            return source_candidate_id
    return None


def _vqa_scene_notes_feedback(vqa_result: VQAResult) -> list[dict[str, Any]] | None:
    if vqa_result.metadata.get("vqa_strategy") != "chronological_scene_notes":
        return None

    notes: list[dict[str, Any]] = []
    for observation in vqa_result.observations:
        if observation.answer is None:
            continue
        notes.append(
            {
                "segment_index": observation.index,
                "start_seconds": observation.start_seconds,
                "end_seconds": observation.end_seconds,
                "notes": observation.answer,
            }
        )
    return notes or None


def _normalized_answer(answer: str) -> str:
    return " ".join(answer.strip().casefold().split())


def _task_with_vqa_answer_feedback(
    task: TaskCreate,
    answer_feedback: list[_VQAAnswerFeedback],
) -> TaskCreate:
    if not answer_feedback:
        return task

    original_description = task.input.description or ""
    feedback_lines = "\n".join(
        f"- {item.answer!r}: {_vqa_answer_feedback_prompt_status(item.status)}"
        f"{_vqa_answer_feedback_segment_suffix(item)}"
        for item in answer_feedback
    )
    scene_notes = _latest_vqa_scene_notes_feedback(answer_feedback)
    scene_notes_block = (
        "\nPrevious VQA scene notes JSON:\n"
        f"{json.dumps(scene_notes, ensure_ascii=False, separators=(',', ':'))}\n"
        if scene_notes
        else ""
    )
    input_payload = task.input.model_copy(
        update={
            "description": (
                f"{original_description}\n\nPrevious VQA answer submissions:\n"
                f"{feedback_lines}\n"
                f"{scene_notes_block}"
                "Do not return an answer that was already submitted. Treat every "
                "previous VQA answer listed above as rejected for retry purposes; "
                "return a different answer only when the video evidence supports "
                "that."
            )
        }
    )
    return task.model_copy(update={"input": input_payload})


def _vqa_answer_feedback_prompt_status(status: str) -> str:
    normalized_status = status.casefold()
    return (
        "rejected"
        if normalized_status in {"rejected", "unavailable"}
        else normalized_status
    )


def _latest_vqa_scene_notes_feedback(
    answer_feedback: list[_VQAAnswerFeedback],
) -> list[dict[str, Any]] | None:
    for item in reversed(answer_feedback):
        if item.scene_notes:
            return item.scene_notes
    return None


def _vqa_answer_feedback_segment_suffix(item: _VQAAnswerFeedback) -> str:
    details: list[str] = []
    if item.segment_index is not None:
        details.append(f"segment_index={item.segment_index!r}")
    if item.start_seconds is not None:
        details.append(f"start_seconds={item.start_seconds!r}")
    if item.end_seconds is not None:
        details.append(f"end_seconds={item.end_seconds!r}")
    if item.source_candidate_id:
        details.append(f"source_candidate_id={item.source_candidate_id!r}")
    return f" [{', '.join(details)}]" if details else ""


def _vqa_searched_video_ids(
    previous_result: RetrievalLoopResult,
) -> set[str]:
    video_ids = set(previous_result.condensed_planner_history.vqa_searched_video_ids)
    for entry in previous_result.full_history:
        if entry.stage != "vqa.video_not_found":
            continue
        request = entry.request if isinstance(entry.request, dict) else {}
        request_video_id = request.get("source_video_id")
        if isinstance(request_video_id, str) and request_video_id:
            video_ids.add(request_video_id)
    return video_ids


def _submitted_vqa_answers(
    previous_result: RetrievalLoopResult,
) -> set[str]:
    answers: set[str] = set()
    for entry in previous_result.full_history:
        if entry.stage != "vqa.answer_submission":
            continue
        request = entry.request if isinstance(entry.request, dict) else {}
        answer = request.get("answer")
        if isinstance(answer, str):
            normalized = _normalized_answer(answer)
            if normalized:
                answers.add(normalized)
    if previous_result.final_answer:
        normalized = _normalized_answer(previous_result.final_answer)
        if normalized:
            answers.add(normalized)
    return answers


def _previous_vqa_answer_feedback(
    previous_result: RetrievalLoopResult,
) -> list[dict[str, Any]]:
    feedback: list[dict[str, Any]] = []
    seen_answers: set[str] = set()
    for entry in previous_result.full_history:
        if entry.stage != "vqa.answer_submission":
            continue
        request = entry.request if isinstance(entry.request, dict) else {}
        response = entry.response if isinstance(entry.response, dict) else {}
        answer = request.get("answer")
        status = response.get("status")
        if not isinstance(answer, str) or not answer:
            continue
        if status == AnswererValidationStatus.ACCEPTED.value:
            continue
        if not isinstance(status, str) or not status:
            continue
        normalized = _normalized_answer(answer)
        if not normalized or normalized in seen_answers:
            continue
        seen_answers.add(normalized)
        feedback.append(
            _VQAAnswerFeedback(
                answer=answer,
                status=status,
                reason=_string_or_none(response.get("reason")),
            ).as_dict()
        )
    return feedback


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _qa_video_not_found_reason(
    source_video_id: str | None,
    detail: str | None,
) -> str:
    video_label = f" video {source_video_id!r}" if source_video_id else " this video"
    detail_text = f" {detail}" if detail else ""
    return (
        f"Final VQA searched{video_label} but did not find a verifiable text "
        f"answer.{detail_text}"
    )


def _image_url(uri: str) -> str:
    """Compatibility helper for callers that imported it from this module."""

    data_layer_settings = _data_layer_settings()
    return urljoin(data_layer_settings["image_path"], normalize_media_uri_path(uri))
