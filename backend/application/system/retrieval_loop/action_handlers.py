"""Handlers for planner actions that do not use the retrieval/browser path."""

from __future__ import annotations

import asyncio
import logging

from application.system.qa import QuestionAnswerer
from application.system.retrieval_loop.candidates import (
    CandidateImageResolver,
    DirectAnswerResolver,
    QAVideoResolver,
    candidate_id,
    candidate_ids,
    direct_answer_candidate,
)
from application.system.retrieval_loop.history import (
    add_timing,
    jsonable,
    llm_history_entries,
    planner_avs_direct_answer_history_step,
    planner_direct_answer_rejected_history_step,
    planner_qa_history_step,
)
from application.system.retrieval_loop.state import (
    IterationContext,
    IterationOutcome,
)
from application.system.retrieval_loop.verification import VerificationPipeline
from application.system.timing import elapsed_ms, timer_start
from application.types.planner.models import ImageQAAction, VideoQAAction
from application.types.qa.models import QAOutput, QARequest
from application.types.retrieval.candidate import Candidate
from application.types.retrieval.queries import DirectAnswer
from application.types.retrieval_loop.models import (
    AVSMatch,
    RetrievalHistoryEntry,
    RetrievalLoopIteration,
)
from application.types.retrieval_loop.evaluation import (
    CandidateEvaluationDecision,
    CandidateEvaluationResult,
)


logger = logging.getLogger(__name__)


class ActionHandlers:
    """Execute QA and direct-answer planner decisions."""

    def __init__(
        self,
        *,
        qa: QuestionAnswerer,
        qa_image_resolver: CandidateImageResolver,
        qa_video_resolver: QAVideoResolver,
        direct_answer_resolver: DirectAnswerResolver,
        verification: VerificationPipeline,
    ) -> None:
        self.qa = qa
        self.qa_image_resolver = qa_image_resolver
        self.qa_video_resolver = qa_video_resolver
        self.direct_answer_resolver = direct_answer_resolver
        self.verification = verification

    async def run_qa(self, context: IterationContext) -> IterationOutcome:
        operation = context.decision.query
        if not isinstance(operation, (ImageQAAction, VideoQAAction)):
            raise TypeError("QA iteration requires an IQA or VQA operation.")

        run = context.run
        materialize_start = timer_start()
        try:
            qa_request = await self._qa_request(context, operation)
        except Exception:
            logger.exception(
                "retrieval_loop.qa.materialize.failed iteration=%s "
                "duration_ms=%.3f action=%r",
                context.index,
                elapsed_ms(materialize_start),
                context.decision.action,
            )
            raise
        materialize_duration = elapsed_ms(materialize_start)
        add_timing(run.timings_ms, "qa.materialize", materialize_duration)
        context.iteration.timings_ms["qa.materialize"] = materialize_duration

        output, qa_duration = await self._run_qa_model(context, qa_request)
        self._record_qa_history(
            context,
            operation,
            qa_request,
            output,
            qa_duration,
            materialize_duration,
        )
        context.finish()
        logger.info(
            "retrieval_loop.qa.done iteration=%s duration_ms=%.3f action=%r",
            context.index,
            qa_duration,
            context.decision.action,
        )
        return IterationOutcome(iteration=context.iteration)

    async def _qa_request(
        self,
        context: IterationContext,
        operation: ImageQAAction | VideoQAAction,
    ) -> QARequest:
        run = context.run
        if isinstance(operation, ImageQAAction):
            candidate = _find_candidate(run.iterations, operation.keyframe_id)
            if candidate is None:
                raise ValueError(
                    "Could not resolve IQA keyframe from prior iterations: "
                    f"{operation.keyframe_id!r}."
                )
            image = await asyncio.to_thread(self.qa_image_resolver, candidate)
            return QARequest(
                question=operation.question,
                image=image,
                dataset=run.task.dataset,
                task_type=run.task.task_type,
            )
        video = await asyncio.to_thread(
            self.qa_video_resolver,
            operation.video_name,
            operation.start_ms,
            operation.end_ms,
            run.task.dataset,
        )
        return QARequest(
            question=operation.question,
            video=video,
            dataset=run.task.dataset,
            task_type=run.task.task_type,
        )

    async def _run_qa_model(
        self,
        context: IterationContext,
        qa_request: QARequest,
    ) -> tuple[QAOutput, float]:
        qa_start = timer_start()
        try:
            output = await self.qa.answer(qa_request)
        except Exception:
            logger.exception(
                "retrieval_loop.qa.failed iteration=%s duration_ms=%.3f action=%r",
                context.index,
                elapsed_ms(qa_start),
                context.decision.action,
            )
            raise
        qa_duration = elapsed_ms(qa_start)
        run = context.run
        add_timing(run.timings_ms, "qa", qa_duration)
        context.iteration.timings_ms["qa"] = qa_duration
        context.iteration.qa_output = output
        return output, qa_duration

    def _record_qa_history(
        self,
        context: IterationContext,
        operation: ImageQAAction | VideoQAAction,
        qa_request: QARequest,
        output: QAOutput,
        qa_duration: float,
        materialize_duration: float,
    ) -> None:
        run = context.run
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="qa",
                kind="module",
                request=jsonable(qa_request),
                response=jsonable(output),
                metadata={
                    "duration_ms": qa_duration,
                    "materialize_duration_ms": materialize_duration,
                    "iteration": context.index,
                    "action": context.decision.action,
                },
            )
        )
        run.full_history.extend(
            llm_history_entries("qa", output.metadata, iteration=context.index)
        )
        run.condensed_history.add_step(
            planner_qa_history_step(
                action=context.decision.action,
                operation=operation,
                output=output,
                purpose=context.decision.purpose,
                policy_rule=context.decision.policy_rule,
            )
        )

    async def run_direct_answer(
        self,
        context: IterationContext,
    ) -> IterationOutcome:
        query = context.decision.query
        if not isinstance(query, DirectAnswer):
            raise TypeError("Direct Answer iteration requires a DirectAnswer query.")

        run = context.run
        candidate = await self._materialize_direct_answer(context, query)

        submission = self.verification.verify(
            context,
            candidate,
            origin="planner_direct_answer",
        )
        if submission.accepted:
            context.finish()
            return IterationOutcome(
                iteration=context.iteration,
                final_candidate=candidate,
                final_submission=submission,
            )

        rejection_reason = submission.reason or (
            "The external submission verifier rejected the submitted frame."
        )

        run.condensed_history.add_step(
            planner_direct_answer_rejected_history_step(
                action=context.decision.action,
                candidate=candidate,
                reason=rejection_reason,
                prior_history=run.condensed_history,
                purpose=context.decision.purpose,
                policy_rule=context.decision.policy_rule,
            )
        )
        context.finish()
        logger.info(
            "retrieval_loop.direct_answer.rejected "
            "iteration=%s candidate_id=%s total_ms=%.3f",
            context.index,
            candidate_ids([candidate])[0],
            context.iteration.timings_ms["total"],
        )
        return IterationOutcome(iteration=context.iteration)

    async def run_avs_direct_answer(
        self,
        context: IterationContext,
    ) -> IterationOutcome:
        """Submit and evaluate a direct AVS proposal without ending early."""

        query = context.decision.query
        if not isinstance(query, DirectAnswer):
            raise TypeError("Direct Answer iteration requires a DirectAnswer query.")

        candidate = await self._materialize_direct_answer(context, query)
        submission = self.verification.verify(
            context,
            candidate,
            origin="planner_direct_answer",
        )
        evaluation = CandidateEvaluationResult(
            candidate_id=candidate_id(candidate),
            matches_task=submission.accepted,
            decision=(
                CandidateEvaluationDecision.ACCEPT
                if submission.accepted
                else CandidateEvaluationDecision.REJECT
            ),
            reasoning=submission.reason,
            metadata={"source": "submission_verifier"},
        )
        context.iteration.evaluator_results.append(evaluation)

        context.run.condensed_history.add_step(
            planner_avs_direct_answer_history_step(
                action=context.decision.action,
                candidate=candidate,
                evaluation=evaluation,
                submission=submission,
                purpose=context.decision.purpose,
                policy_rule=context.decision.policy_rule,
            )
        )
        context.finish()
        logger.info(
            "retrieval_loop.avs.direct_answer.done iteration=%s candidate_id=%s "
            "accepted=%s submission_status=%s total_ms=%.3f",
            context.index,
            candidate_id(candidate),
            submission.accepted,
            submission.status,
            context.iteration.timings_ms["total"],
        )
        return IterationOutcome(
            iteration=context.iteration,
            avs_matches=(
                (AVSMatch(candidate=candidate, evaluation=evaluation),)
                if submission.accepted
                else ()
            ),
        )

    async def _materialize_direct_answer(
        self,
        context: IterationContext,
        query: DirectAnswer,
    ) -> Candidate:
        run = context.run
        materialize_start = timer_start()
        request = {
            "video_id": query.videoId,
            "timestamp_ms": query.timestampMs,
            "dataset": run.task.dataset.value,
        }
        try:
            logger.info(
                "retrieval_loop.direct_answer.materialize.start "
                "iteration=%s video_id=%s timestamp_ms=%s",
                context.index,
                query.videoId,
                query.timestampMs,
            )
            if self.direct_answer_resolver is direct_answer_candidate:
                candidate = await asyncio.to_thread(
                    self.direct_answer_resolver,
                    query.videoId,
                    query.timestampMs,
                    run.task.dataset,
                )
            else:
                candidate = self.direct_answer_resolver(
                    query.videoId,
                    query.timestampMs,
                    run.task.dataset,
                )
        except Exception:
            logger.exception(
                "retrieval_loop.direct_answer.materialize.failed "
                "iteration=%s duration_ms=%.3f video_id=%s timestamp_ms=%s",
                context.index,
                elapsed_ms(materialize_start),
                query.videoId,
                query.timestampMs,
            )
            raise

        materialize_duration = elapsed_ms(materialize_start)
        add_timing(run.timings_ms, "direct_answer.materialize", materialize_duration)
        context.iteration.timings_ms["direct_answer.materialize"] = materialize_duration
        context.iteration.candidates = [candidate]
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="direct_answer.materialize",
                kind="module",
                request=request,
                response=jsonable(candidate),
                metadata={
                    "duration_ms": materialize_duration,
                    "iteration": context.index,
                },
            )
        )
        logger.info(
            "retrieval_loop.direct_answer.materialize.done "
            "iteration=%s duration_ms=%.3f candidate_id=%s",
            context.index,
            materialize_duration,
            candidate_ids([candidate])[0],
        )
        return candidate


def _find_candidate(
    iterations: list[RetrievalLoopIteration],
    expected_id: str,
) -> Candidate | None:
    for iteration in reversed(iterations):
        for candidate in iteration.candidates:
            if candidate_id(candidate) == expected_id:
                return candidate
    return None
