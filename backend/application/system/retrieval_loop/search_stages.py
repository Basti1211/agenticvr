"""Planner and data-layer retrieval stages for the retrieval loop."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import HTTPException

from application.settings.settings import _data_layer_settings
from application.system.planner import Planner
from application.tasks.models import TaskType
from application.system.retrieval.candidate_filters import (
    apply_candidate_filters,
    competition_candidate_filters,
    scoped_candidate_filters,
)
from application.system.retrieval_loop.candidates import (
    CandidateImageResolver,
    candidate_id,
    candidate_ids,
    candidate_image_input,
    candidate_image_inputs,
)
from application.system.retrieval_loop.history import (
    add_timing,
    jsonable,
    llm_history_entries,
    planner_failed_history_step,
    planner_no_candidates_history_step,
    planner_retry_history_step,
)
from application.system.retrieval_loop.retrieval_adapter import (
    retrieval_dataset_name,
)
from application.system.retrieval_loop.state import IterationContext, RunContext
from application.system.task_input_access import (
    DEFAULT_TASK_INPUT_ACCESS,
    ModuleTaskInputAccess,
    encoded_task_for_module,
    task_videos_for_module,
)
from application.system.timing import elapsed_ms, timer_start
from application.types.planner.models import PlannerCondensedHistory, PlannerRequest
from application.types.retrieval.candidate import Candidate
from application.types.retrieval_loop.models import (
    RetrievalHistoryEntry,
    RetrievalLoopIteration,
)
from application.types.vlm.prompt import ImageInput


Retriever = Callable[[str, str], Awaitable[list[Candidate]]]
BrowserRetriever = Callable[
    [str, str, set[str], int],
    Awaitable[list[Candidate]],
]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedCandidates:
    candidates: list[Candidate]
    duration_ms: float
    request: dict[str, str]


class SearchStages:
    """Execute planner and retrieval stages with their retry semantics."""

    def __init__(
        self,
        *,
        planner: Planner,
        retriever: Retriever,
        browser_retriever: BrowserRetriever | None = None,
        planner_image_resolver: CandidateImageResolver | None = None,
        task_input_access: ModuleTaskInputAccess | None = None,
        task_reference_video_resolver=None,
        hide_avs_submission_ratings_from_planner: bool = False,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.browser_retriever = browser_retriever
        self.planner_image_resolver = planner_image_resolver
        self.include_keyframe_references = planner.keyframe_references_enabled
        self.task_input_access = task_input_access or DEFAULT_TASK_INPUT_ACCESS.planner
        self.task_reference_video_resolver = task_reference_video_resolver
        self.hide_avs_submission_ratings_from_planner = (
            hide_avs_submission_ratings_from_planner
        )

    async def plan(
        self,
        run: RunContext,
        index: int,
        iteration_start: float,
    ) -> IterationContext | RetrievalLoopIteration:
        planner_start = timer_start()
        logger.info("retrieval_loop.planner.start iteration=%s", index)
        try:
            history_images = await self._history_images(run)
            planner_request = PlannerRequest(
                encoded_task=encoded_task_for_module(
                    run.encoded,
                    run.task,
                    self.task_input_access,
                ),
                dataset=run.task.dataset,
                task_type=run.task.task_type,
                history=self._planner_history(run),
                history_images=history_images,
                submission_ratings_visible=(
                    not self.hide_avs_submission_ratings_from_planner
                    or run.task.task_type != TaskType.AVS
                ),
                task_videos=(
                    task_videos_for_module(
                        run.task,
                        self.task_input_access,
                        self.task_reference_video_resolver,
                    )
                    if self.task_reference_video_resolver is not None
                    else []
                ),
            )
        except Exception:
            logger.exception(
                "retrieval_loop.planner.materialize.failed iteration=%s "
                "duration_ms=%.3f",
                index,
                elapsed_ms(planner_start),
            )
            raise

        try:
            decision = await self.planner.plan(planner_request)
        except ValueError as error:
            return self._record_planner_retryable_error(
                run,
                index,
                error,
                planner_start,
                iteration_start,
                planner_request,
            )
        except Exception:
            logger.exception(
                "retrieval_loop.planner.failed iteration=%s duration_ms=%.3f",
                index,
                elapsed_ms(planner_start),
            )
            raise

        planner_duration = elapsed_ms(planner_start)
        add_timing(run.timings_ms, "planner", planner_duration)
        logger.info(
            "retrieval_loop.planner.done iteration=%s duration_ms=%.3f "
            "endpoint=%s action=%r",
            index,
            planner_duration,
            decision.endpoint,
            decision.action,
        )
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="planner",
                kind="module",
                request=jsonable(planner_request),
                response=jsonable(decision),
                metadata={"duration_ms": planner_duration, "iteration": index},
            )
        )
        run.full_history.extend(
            llm_history_entries("planner", decision.metadata, iteration=index)
        )
        iteration = RetrievalLoopIteration(
            index=index,
            action=decision.action,
            planner_purpose=decision.purpose,
            planner_policy_rule=decision.policy_rule,
            endpoint=decision.endpoint,
        )
        iteration.timings_ms["planner"] = planner_duration
        return IterationContext(
            run=run,
            decision=decision,
            iteration=iteration,
            started_at=iteration_start,
        )

    async def _history_images(self, run: RunContext) -> list[ImageInput]:
        """Materialize the latest rated retrieval keyframes for the planner."""

        if not self.include_keyframe_references:
            return []

        def materialize() -> list[ImageInput]:
            images: list[ImageInput] = []
            submitted_ids = self._submitted_candidate_ids(run)
            for candidate in run.planner_reference_candidates:
                if candidate_id(candidate) in submitted_ids:
                    continue
                if candidate.frames:
                    images.extend(candidate_image_inputs(candidate))
                elif self.planner_image_resolver is not None:
                    images.append(self.planner_image_resolver(candidate))
                else:
                    images.append(candidate_image_input(candidate))
            return images

        return await asyncio.to_thread(materialize)

    def _planner_history(self, run: RunContext) -> PlannerCondensedHistory:
        history = run.condensed_history.model_copy(deep=True)
        if (
            not self.hide_avs_submission_ratings_from_planner
            or run.task.task_type != TaskType.AVS
        ):
            return history

        submitted_ids = {
            item.candidate_id
            for item in history.submitted_candidates
            if item.candidate_id is not None
        }
        for item in history.submitted_candidates:
            item.status = None
        history.wrongly_submitted_candidates = []
        history.keyframe_evidence = [
            item
            for item in history.keyframe_evidence
            if item.keyframe_id not in submitted_ids
        ]

        actions = list(history.used_actions)
        if history.latest_valid_action is not None:
            actions.append(history.latest_valid_action)
        for action in actions:
            action_ids = set(action.result_candidate_ids or [])
            rated_ids = action_ids & submitted_ids
            if not rated_ids:
                continue
            action.result_candidate_ids = [
                item
                for item in action.result_candidate_ids or []
                if item not in submitted_ids
            ]
            action.result_groups.relevant = [
                item
                for item in action.result_groups.relevant
                if item not in submitted_ids
            ]
            action.result_groups.highly_relevant = [
                item
                for item in action.result_groups.highly_relevant
                if item not in submitted_ids
            ]
            action.result_groups.irrelevant = [
                item
                for item in action.result_groups.irrelevant
                if item not in submitted_ids
            ]
            action.result_summary = None
            action.evaluator_feedback = None
            action.irrelevant_reason = None
        return history

    def _submitted_candidate_ids(self, run: RunContext) -> set[str]:
        if (
            not self.hide_avs_submission_ratings_from_planner
            or run.task.task_type != TaskType.AVS
        ):
            return set()
        return {
            item.candidate_id
            for item in run.condensed_history.submitted_candidates
            if item.candidate_id is not None
        }

    async def retrieve(
        self,
        context: IterationContext,
        *,
        candidate_limit: int,
    ) -> RetrievedCandidates | None:
        run = context.run
        retrieval_request = {
            "action": context.decision.action,
            "dataset": run.task.dataset.value,
            "endpoint": context.decision.endpoint,
        }
        retrieval_start = timer_start()
        try:
            logger.info(
                "retrieval_loop.retrieval.start iteration=%s endpoint=%s "
                "action=%r dataset=%s",
                context.index,
                context.decision.endpoint,
                context.decision.action,
                run.task.dataset.value,
            )
            dataset = retrieval_dataset_name(run.task.dataset.value)
            task_filters = competition_candidate_filters(
                dataset=run.task.dataset.value,
                source_task_name=run.task.source_task_name,
                rules=_competition_candidate_filter_rules(),
            )
            with scoped_candidate_filters(task_filters):
                if self.browser_retriever is None:
                    candidates = await self.retriever(
                        context.decision.action,
                        dataset,
                    )
                else:
                    # `--unseen` backfilling must also skip candidates rejected by
                    # the browser. Iterative hint transitions intentionally clear
                    # `seen_candidate_ids` while retaining those rejections; if we
                    # pass only the seen set, the retriever fills its page with
                    # candidates that the loop discards immediately afterwards.
                    excluded_candidate_ids = (
                        run.seen_candidate_ids | run.irrelevant_candidate_ids
                    )
                    candidates = await self.browser_retriever(
                        context.decision.action,
                        dataset,
                        set(excluded_candidate_ids),
                        candidate_limit,
                    )
            if task_filters:
                candidates = apply_candidate_filters(candidates, task_filters)
        except (HTTPException, ValueError) as error:
            self._record_retrieval_retryable_error(
                context,
                error,
                retrieval_start,
                retrieval_request,
            )
            return None
        except Exception:
            logger.exception(
                "retrieval_loop.retrieval.failed iteration=%s endpoint=%s "
                "action=%r dataset=%s duration_ms=%.3f",
                context.index,
                context.decision.endpoint,
                context.decision.action,
                run.task.dataset.value,
                elapsed_ms(retrieval_start),
            )
            raise

        retrieval_duration = elapsed_ms(retrieval_start)
        add_timing(run.timings_ms, "retrieval", retrieval_duration)
        context.iteration.timings_ms["retrieval"] = retrieval_duration
        return RetrievedCandidates(
            candidates=candidates,
            duration_ms=retrieval_duration,
            request=retrieval_request,
        )

    def record_success(
        self,
        context: IterationContext,
        retrieved: RetrievedCandidates,
        candidates: list[Candidate],
    ) -> None:
        logger.info(
            "retrieval_loop.retrieval.done iteration=%s duration_ms=%.3f "
            "retrieved_candidate_count=%s used_candidate_count=%s "
            "candidate_ids=%s",
            context.index,
            retrieved.duration_ms,
            len(retrieved.candidates),
            len(candidates),
            candidate_ids(candidates),
        )
        context.run.full_history.append(
            RetrievalHistoryEntry(
                stage="retrieval",
                kind="retrieval",
                request=retrieved.request,
                response=jsonable(retrieved.candidates),
                metadata={
                    "duration_ms": retrieved.duration_ms,
                    "iteration": context.index,
                    "action": context.decision.action,
                    "retrieved_candidate_count": len(retrieved.candidates),
                    "used_candidate_count": len(candidates),
                },
            )
        )

    def finish_empty(self, context: IterationContext) -> None:
        context.finish()
        logger.info(
            "retrieval_loop.iteration.no_candidates iteration=%s total_ms=%.3f",
            context.index,
            context.iteration.timings_ms["total"],
        )
        context.run.condensed_history.add_step(
            planner_no_candidates_history_step(
                context.decision.action,
                purpose=context.decision.purpose,
                policy_rule=context.decision.policy_rule,
            )
        )
        filtered_candidate_ids = candidate_ids(context.iteration.filtered_candidates)
        if filtered_candidate_ids:
            context.run.condensed_history.include_seen_keyframes(
                filtered_candidate_ids,
                returned_candidate_count=len(context.iteration.filtered_candidates),
                new_candidate_count=0,
            )

    def _record_planner_retryable_error(
        self,
        run: RunContext,
        index: int,
        error: ValueError,
        planner_start: float,
        iteration_start: float,
        planner_request: PlannerRequest,
    ) -> RetrievalLoopIteration:
        planner_duration = elapsed_ms(planner_start)
        add_timing(run.timings_ms, "planner", planner_duration)
        planner_error = str(error)
        logger.warning(
            "retrieval_loop.planner.retryable_failed iteration=%s "
            "duration_ms=%.3f error=%s",
            index,
            planner_duration,
            planner_error,
        )
        iteration = RetrievalLoopIteration(
            index=index,
            action="planner_error",
            endpoint="planner",
            retrieval_error=planner_error,
        )
        iteration.timings_ms["planner"] = planner_duration
        iteration.timings_ms["total"] = elapsed_ms(iteration_start)
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="planner",
                kind="module",
                request=jsonable(planner_request),
                error=planner_error,
                metadata={"duration_ms": planner_duration, "iteration": index},
            )
        )
        run.condensed_history.add_step(planner_failed_history_step(error=planner_error))
        return iteration

    def _record_retrieval_retryable_error(
        self,
        context: IterationContext,
        error: HTTPException | ValueError,
        retrieval_start: float,
        retrieval_request: dict[str, str],
    ) -> None:
        run = context.run
        retrieval_duration = elapsed_ms(retrieval_start)
        add_timing(run.timings_ms, "retrieval", retrieval_duration)
        context.iteration.timings_ms["retrieval"] = retrieval_duration
        context.finish()
        context.iteration.retrieval_error = _retryable_retrieval_error(error)
        logger.warning(
            "retrieval_loop.retrieval.retryable_failed iteration=%s "
            "duration_ms=%.3f endpoint=%s action=%r error=%s",
            context.index,
            retrieval_duration,
            context.decision.endpoint,
            context.decision.action,
            context.iteration.retrieval_error,
        )
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="retrieval",
                kind="retrieval",
                request=retrieval_request,
                error=context.iteration.retrieval_error,
                metadata={
                    "duration_ms": retrieval_duration,
                    "iteration": context.index,
                    "action": context.decision.action,
                },
            )
        )
        run.condensed_history.add_step(
            planner_retry_history_step(
                action=context.decision.action,
                error=context.iteration.retrieval_error,
                purpose=context.decision.purpose,
                policy_rule=context.decision.policy_rule,
            )
        )


def _retryable_retrieval_error(error: HTTPException | ValueError) -> str:
    if isinstance(error, HTTPException):
        return str(error.detail)
    return str(error)


def _competition_candidate_filter_rules() -> list[dict]:
    rules = _data_layer_settings().get("competition_candidate_filters", [])
    if not isinstance(rules, list):
        raise ValueError("Data-layer competition_candidate_filters must be a list.")
    return [rule for rule in rules if isinstance(rule, dict)]
