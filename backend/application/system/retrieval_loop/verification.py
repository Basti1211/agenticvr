"""External submission verification for browser-exact and direct candidates."""

from __future__ import annotations

import logging

from application.system.retrieval_loop.history import add_timing, jsonable
from application.system.retrieval_loop.state import IterationContext
from application.system.retrieval_loop.submission import verify_submission
from application.system.timing import elapsed_ms, timer_start
from application.tasks.models import TaskType
from application.types.retrieval.candidate import Candidate
from application.types.retrieval_loop.models import (
    CandidateSubmission,
    CandidateSubmissionStatus,
    RetrievalHistoryEntry,
)


logger = logging.getLogger(__name__)
NEAR_MATCH_HISTORY_REASON = (
    "This visually plausible candidate was rejected as the exact target. Treat its "
    "local scene interval as negative evidence. The rejection does not imply that "
    "the target occurs elsewhere in the same video; search globally unless "
    "independent evidence identifies a distinct scene."
)


class VerificationPipeline:
    """Submit authoritative browser matches to the external verifier."""

    def verify(
        self,
        context: IterationContext,
        candidate: Candidate,
        *,
        origin: str = "browser_scene",
    ) -> CandidateSubmission:
        run = context.run
        if run.task.task_type == TaskType.QA:
            raise RuntimeError(
                "QA tasks must not submit image or video candidates; browser-exact "
                "scenes must be sent to VQA."
            )
        submission_start = timer_start()
        submission = verify_submission(task=run.task, candidate=candidate)
        if submission.source == "unverified":
            submission = submission.model_copy(
                update={
                    "source": origin,
                    "reason": (
                        "Accepted for submission because the authoritative browser "
                        "scene classifier found an exact match and no explicit "
                        "target was available."
                        if origin == "browser_scene"
                        else "Accepted because no explicit target was available to "
                        "externally check the planner's direct answer."
                    ),
                }
            )
        if (
            not submission.accepted
            and submission.status != CandidateSubmissionStatus.UNAVAILABLE
            and submission.source != "competition_judgement"
        ):
            submission = submission.model_copy(
                update={"reason": NEAR_MATCH_HISTORY_REASON}
            )
        duration = elapsed_ms(submission_start)
        add_timing(run.timings_ms, "submission_verifier", duration)
        context.iteration.timings_ms["submission_verifier"] = (
            context.iteration.timings_ms.get("submission_verifier", 0.0) + duration
        )
        context.iteration.submissions.append(submission)
        run.full_history.append(
            RetrievalHistoryEntry(
                stage="submission_verifier",
                kind="module",
                request={
                    "candidate_id": submission.candidate_id,
                    "origin": origin,
                    "has_answer_video": run.task.input.answer_video is not None,
                    "has_competition_judgement": (
                        submission.source == "competition_judgement"
                    ),
                },
                response=jsonable(submission),
                metadata={"duration_ms": duration, "iteration": context.index},
            )
        )
        logger.info(
            "retrieval_loop.submission_verifier.done iteration=%s "
            "duration_ms=%.3f candidate_id=%s accepted=%s source=%s",
            context.index,
            duration,
            submission.candidate_id,
            submission.accepted,
            submission.source,
        )
        return submission
