"""History, timing, and planner-feedback helpers for retrieval loops."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from application.system.retrieval_loop.candidates import candidate_id
from application.types.browser.models import (
    BrowserClassificationOutput,
    BrowserDescriptionOutput,
    BrowserReasoningOutput,
)
from application.types.evidence import TemporalStatus, VisualSimilarity
from application.types.planner.models import (
    ImageQAAction,
    PlannerCondensedHistory,
    PlannerEvaluatedImage,
    PlannerEvaluatorResult,
    PlannerHistoryStep,
    PlannerIrrelevantImageGroup,
    PlannerQAResult,
    PlannerSubmittedCandidate,
    PlannerWatchedSegment,
    PlannerWronglySubmittedCandidate,
    VideoQAAction,
)
from application.types.qa.models import QAOutput
from application.types.retrieval.candidate import Candidate
from application.types.retrieval_loop.models import (
    CandidateSubmission,
    RetrievalHistoryEntry,
)
from application.types.retrieval_loop.evaluation import (
    CandidateEvaluationResult,
)


def planner_retry_history_step(
    action: str,
    error: str,
    *,
    purpose: str | None = None,
    policy_rule: str | None = None,
) -> PlannerHistoryStep:
    return PlannerHistoryStep(
        action=action,
        purpose=purpose,
        policy_rule=policy_rule,
        action_valid=False,
        result_summary=f"This action was invalid: {error}",
        evaluator_result=PlannerEvaluatorResult(
            evaluator_feedback=(
                f"The data layer could not evaluate this action: {error}. Choose a "
                "different valid action using the exact allowed syntax and only ids or "
                "values present in the provided evidence."
            ),
        ),
    )


def planner_no_candidates_history_step(
    action: str,
    *,
    purpose: str | None = None,
    policy_rule: str | None = None,
) -> PlannerHistoryStep:
    return PlannerHistoryStep(
        action=action,
        purpose=purpose,
        policy_rule=policy_rule,
        action_valid=True,
        result_summary="Out of 0 keyframes, 0 were relevant and 0 were irrelevant.",
        evaluator_result=PlannerEvaluatorResult(
            evaluator_feedback=(
                "No candidates were found. Use a different valid action or broaden only "
                "the least distinctive constraint while preserving the target's core subject "
                "and action."
            ),
        ),
    )


def planner_failed_history_step(error: str) -> PlannerHistoryStep:
    return PlannerHistoryStep(
        action="planner_error",
        action_valid=False,
        result_summary=f"The previous action output was invalid: {error}",
        evaluator_result=PlannerEvaluatorResult(
            evaluator_feedback=(
                "The previous planner attempts were invalid: "
                f"{error}. Choose one valid action using the exact allowed syntax. Do "
                "not invent unsupported ids or parameters. Repeating an earlier action "
                "is allowed."
            ),
        ),
    )


def planner_direct_answer_rejected_history_step(
    *,
    action: str,
    candidate: Candidate,
    reason: str,
    prior_history: PlannerCondensedHistory | None = None,
    purpose: str | None = None,
    policy_rule: str | None = None,
) -> PlannerHistoryStep:
    submitted_id = candidate_id(candidate)
    rejected_interval = _rejected_scene_interval(prior_history, candidate)
    watched_segments = (
        [
            PlannerWatchedSegment(
                candidate_id=submitted_id,
                start_ms=rejected_interval[0],
                end_ms=rejected_interval[1],
            )
        ]
        if rejected_interval is not None
        else None
    )
    return PlannerHistoryStep(
        action=action,
        purpose=purpose,
        policy_rule=policy_rule,
        action_valid=True,
        result_candidate_ids=[submitted_id],
        result_summary=(
            f"The directly submitted frame {submitted_id} was evaluated and rejected."
        ),
        watched_segments=watched_segments,
        evaluator_result=PlannerEvaluatorResult(
            highly_relevant_images=[
                PlannerEvaluatedImage(
                    image_id=submitted_id,
                    description=_candidate_history_description(candidate),
                    reason=reason,
                    video_id=(
                        "_".join(candidate.id[:-1]) if len(candidate.id) > 1 else None
                    ),
                    timestamp_ms=candidate.time_keyframe,
                )
            ],
            evaluator_feedback=(
                "The direct answer was not accepted. Treat the submitted local scene "
                "as negative evidence. Return to global retrieval unless independent "
                "evidence identifies a distinct scene; do not walk through adjacent "
                "timestamps from the rejected scene."
            ),
        ),
        wrongly_submitted_candidates=[
            _wrongly_submitted_candidate(
                candidate,
                scene_interval=rejected_interval,
            )
        ],
    )


def planner_avs_direct_answer_history_step(
    *,
    action: str,
    candidate: Candidate,
    evaluation: CandidateEvaluationResult,
    submission: CandidateSubmission,
    purpose: str | None = None,
    policy_rule: str | None = None,
) -> PlannerHistoryStep:
    """Return AVS evaluator feedback for a directly proposed candidate."""

    submitted_id = candidate_id(candidate)
    image = PlannerEvaluatedImage(
        image_id=submitted_id,
        description=_candidate_history_description(candidate),
        video_id=("_".join(candidate.id[:-1]) if len(candidate.id) > 1 else None),
        timestamp_ms=candidate.time_keyframe,
        reason=(
            evaluation.reasoning
            or (
                "The browser found an exact AVS scene and verification accepted it."
                if evaluation.matches_task
                else "Submission verification rejected this AVS candidate."
            )
        ),
        visual_similarity=(
            evaluation.visual_similarity
            if evaluation.visual_similarity != VisualSimilarity.UNKNOWN
            else None
        ),
        temporal_status=(
            evaluation.temporal_status
            if evaluation.temporal_status != TemporalStatus.NOT_APPLICABLE
            else None
        ),
    )
    decision = submission.status.value
    accepted = submission.accepted
    return PlannerHistoryStep(
        action=action,
        purpose=purpose,
        policy_rule=policy_rule,
        action_valid=True,
        result_candidate_ids=[submitted_id],
        result_summary=(
            f"The submitted AVS candidate {submitted_id} was classified as {decision}."
        ),
        evaluator_result=PlannerEvaluatorResult(
            relevant_images=[image] if accepted else [],
            highly_relevant_images=[] if accepted else [image],
            evaluator_feedback=(
                f"AVS submission checker classified candidate {submitted_id!r} "
                f"as {decision}. Continue searching for additional distinct matches."
            ),
        ),
        submitted_candidates=[_submitted_candidate(candidate, submission)],
        wrongly_submitted_candidates=(
            [_wrongly_submitted_candidate(candidate)]
            if submission.status.value == "rejected"
            else []
        ),
    )


def planner_qa_history_step(
    *,
    action: str,
    operation: ImageQAAction | VideoQAAction,
    output: QAOutput,
    purpose: str | None = None,
    policy_rule: str | None = None,
) -> PlannerHistoryStep:
    """Return structured QA evidence for the next planner iteration."""

    if isinstance(operation, ImageQAAction):
        media_id = operation.keyframe_id
        start_ms = None
        end_ms = None
        candidate_ids = [media_id]
        evidence = f"keyframe {media_id}"
    else:
        media_id = operation.video_name
        start_ms = operation.start_ms
        end_ms = operation.end_ms
        candidate_ids = []
        evidence = (
            f"video {media_id} from {operation.start_ms} ms to {operation.end_ms} ms"
        )

    return PlannerHistoryStep(
        action=action,
        purpose=purpose,
        policy_rule=policy_rule,
        action_valid=True,
        result_candidate_ids=candidate_ids,
        result_summary=f"QA inspected {evidence} and returned a textual answer.",
        qa_result=PlannerQAResult(
            media_id=media_id,
            start_ms=start_ms,
            end_ms=end_ms,
            question=operation.question,
            answer=output.answer,
        ),
    )


def planner_feedback_history_step(
    *,
    action: str,
    purpose: str | None = None,
    policy_rule: str | None = None,
    candidates_by_id: dict[str, Candidate],
    browser_output: BrowserClassificationOutput,
    browser_reasoning: BrowserReasoningOutput,
    evaluator_results: list[CandidateEvaluationResult],
    browser_descriptions: BrowserDescriptionOutput | None = None,
    inspected_video_candidate_ids: set[str] | None = None,
    inspected_video_intervals_ms: dict[str, tuple[int, int]] | None = None,
    submissions: list[CandidateSubmission] | None = None,
    include_empty_feedback: bool = True,
    include_accepted_evaluations: bool = False,
) -> PlannerHistoryStep:
    rejected_submission_by_id = {
        submission.candidate_id: submission
        for submission in submissions or []
        if (
            not submission.accepted
            and submission.status != "unavailable"
        )
    }
    unavailable_submission_by_id = {
        submission.candidate_id: submission
        for submission in submissions or []
        if submission.status == "unavailable"
    }
    rejection_result_by_id = {
        result.candidate_id: result
        for result in evaluator_results
        if not result.matches_task and result.candidate_id in candidates_by_id
    }
    accepted_result_by_id = {
        result.candidate_id: result
        for result in evaluator_results
        if (
            result.matches_task
            and result.candidate_id in candidates_by_id
            and result.candidate_id not in rejected_submission_by_id
            and result.candidate_id not in unavailable_submission_by_id
        )
    }
    rejected_candidate_ids = _unique(
        [
            result.candidate_id
            for result in evaluator_results
            if (
                (
                    not result.matches_task
                    or result.candidate_id in rejected_submission_by_id
                    or result.candidate_id in unavailable_submission_by_id
                )
                and result.candidate_id in candidates_by_id
            )
        ]
        + [
            candidate_id
            for candidate_id in rejected_submission_by_id
            if candidate_id in candidates_by_id
        ]
        + [
            candidate_id
            for candidate_id in unavailable_submission_by_id
            if candidate_id in candidates_by_id
        ]
    )
    not_match_images = [
        _planner_evaluated_image(
            candidate_id,
            candidates_by_id[candidate_id],
            browser_descriptions,
            evaluation=rejection_result_by_id.get(candidate_id),
            reason=_rejection_reason(
                candidate_id,
                rejection_result_by_id,
                rejected_submission_by_id,
                unavailable_submission_by_id,
            ),
        )
        for candidate_id in rejected_candidate_ids
    ]

    relevant_reason_by_id = {
        candidate_id: reasoning.reasoning
        for reasoning in browser_reasoning.relevant
        for candidate_id in reasoning.candidate_ids
        if candidate_id in candidates_by_id
    }
    relevant_ids = _unique(
        [
            candidate_id
            for candidate_id in browser_output.relevant_candidate_ids
            if candidate_id in candidates_by_id
        ]
        + [
            candidate_id
            for candidate_id in browser_output.match_candidate_ids
            if candidate_id in candidates_by_id
            and candidate_id not in rejected_candidate_ids
        ]
        + (list(accepted_result_by_id) if include_accepted_evaluations else [])
    )
    relevant_images = [
        _planner_evaluated_image(
            candidate_id,
            candidates_by_id[candidate_id],
            browser_descriptions,
            evaluation=accepted_result_by_id.get(candidate_id),
            reason=(
                _acceptance_reason(accepted_result_by_id[candidate_id])
                if candidate_id in accepted_result_by_id
                else _browser_relevance_reason(
                    candidate_id,
                    browser_descriptions,
                    relevant_reason_by_id,
                )
            ),
        )
        for candidate_id in relevant_ids
    ]
    irrelevant_ids = _unique(
        [
            candidate_id
            for candidate_id in browser_output.irrelevant_candidate_ids
            if candidate_id in candidates_by_id
        ]
    )
    feedback = (
        _avs_match_feedback(evaluator_results)
        if include_accepted_evaluations and evaluator_results
        else None
    )
    # Kept in the signature for callers that still pass the legacy option. Empty
    # result pages are now represented by the compact count summary below.
    _ = include_empty_feedback

    retained_ids = _unique(
        [image.image_id for image in relevant_images]
        + [image.image_id for image in not_match_images]
    )

    return PlannerHistoryStep(
        action=action,
        purpose=purpose,
        policy_rule=policy_rule,
        action_valid=True,
        result_candidate_ids=retained_ids,
        result_summary=_candidate_result_summary(
            candidate_count=len(candidates_by_id),
            relevant_count=len(retained_ids),
        ),
        watched_segments=_watched_segments(
            candidates_by_id,
            (
                set(candidates_by_id)
                if inspected_video_candidate_ids is None
                else inspected_video_candidate_ids
            )
            & set(retained_ids),
            inspected_video_intervals_ms=inspected_video_intervals_ms,
        ),
        evaluator_result=PlannerEvaluatorResult(
            relevant_images=relevant_images,
            highly_relevant_images=not_match_images,
            irrelevant_groups=(
                [PlannerIrrelevantImageGroup(image_ids=irrelevant_ids)]
                if irrelevant_ids
                else []
            ),
            evaluator_feedback=feedback,
            irrelevant_reason=(
                browser_reasoning.irrelevant.reasoning
                if browser_reasoning.irrelevant is not None
                else None
            ),
        ),
        wrongly_submitted_candidates=[
            _wrongly_submitted_candidate(
                candidates_by_id[candidate_id],
                scene_interval=(inspected_video_intervals_ms or {}).get(candidate_id),
            )
            for candidate_id in rejected_submission_by_id
            if candidate_id in candidates_by_id
        ],
        submitted_candidates=[
            _submitted_candidate(candidates_by_id[submission.candidate_id], submission)
            for submission in submissions or []
            if submission.status is not None
            and submission.candidate_id in candidates_by_id
        ],
    )


def has_planner_feedback(step: PlannerHistoryStep) -> bool:
    result = step.evaluator_result
    return bool(
        result.relevant_images
        or result.highly_relevant_images
        or result.irrelevant_groups
        or result.evaluator_feedback
    )


def evaluation_llm_calls(
    evaluations: list[CandidateEvaluationResult],
) -> list[dict[str, Any]]:
    return [
        call
        for evaluation in evaluations
        for call in evaluation.metadata.get("llm_calls", [])
        if isinstance(call, dict)
    ]


def llm_history_entries(
    stage: str,
    metadata: dict[str, Any],
    iteration: int | None = None,
) -> list[RetrievalHistoryEntry]:
    return [
        RetrievalHistoryEntry(
            stage=stage,
            kind="llm",
            request=call.get("prompt"),
            response=call.get("response"),
            error=call.get("error"),
            metadata={
                key: value
                for key, value in call.items()
                if key not in {"prompt", "response", "error"}
            }
            | ({"iteration": iteration} if iteration is not None else {}),
        )
        for call in metadata.get("llm_calls", metadata.get("attempts", []))
        if isinstance(call, dict)
    ]


def add_timing(timings_ms: dict[str, float], key: str, duration_ms: float) -> None:
    timings_ms[key] = round(timings_ms.get(key, 0.0) + duration_ms, 3)


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def _candidate_history_description(candidate: Candidate) -> str:
    video_id = "_".join(candidate.id[:-1]) if len(candidate.id) > 1 else None
    video_label = f" from video {video_id}" if video_id else ""
    return f"Keyframe{video_label} at {candidate.time_keyframe} ms."


def _wrongly_submitted_candidate(
    candidate: Candidate,
    *,
    scene_interval: tuple[int, int] | None = None,
) -> PlannerWronglySubmittedCandidate:
    video_id = "_".join(candidate.id[:-1]) if len(candidate.id) > 1 else candidate.id[0]
    if scene_interval is None and candidate.time_scene_end > candidate.time_scene_start:
        scene_interval = (candidate.time_scene_start, candidate.time_scene_end)
    return PlannerWronglySubmittedCandidate(
        video_id=video_id,
        timestamp_ms=candidate.time_keyframe,
        scene_start_ms=scene_interval[0] if scene_interval is not None else None,
        scene_end_ms=scene_interval[1] if scene_interval is not None else None,
    )


def _rejected_scene_interval(
    history: PlannerCondensedHistory | None,
    candidate: Candidate,
) -> tuple[int, int] | None:
    """Resolve the narrowest inspected scene containing a direct-answer timestamp."""

    candidate_video_id = (
        "_".join(candidate.id[:-1]) if len(candidate.id) > 1 else candidate.id[0]
    )
    matching_intervals: list[tuple[int, int]] = []
    if history is not None:
        for evidence in history.keyframe_evidence:
            segment = evidence.inspected_segment
            if segment is None or evidence.video_id is None:
                continue
            if evidence.video_id.casefold() != candidate_video_id.casefold():
                continue
            if segment.start_ms <= candidate.time_keyframe <= segment.end_ms:
                matching_intervals.append((segment.start_ms, segment.end_ms))
    if matching_intervals:
        return min(matching_intervals, key=lambda interval: interval[1] - interval[0])
    if candidate.time_scene_end > candidate.time_scene_start:
        return (candidate.time_scene_start, candidate.time_scene_end)
    return None


def _submitted_candidate(
    candidate: Candidate,
    submission: CandidateSubmission,
) -> PlannerSubmittedCandidate:
    video_id = submission.video_id or (
        "_".join(candidate.id[:-1]) if len(candidate.id) > 1 else candidate.id[0]
    )
    return PlannerSubmittedCandidate(
        video_id=video_id,
        timestamp_ms=submission.timestamp_ms or candidate.time_keyframe,
        status=submission.status.value,
        candidate_id=submission.candidate_id,
        scene_start_ms=submission.scene_start_ms,
        scene_end_ms=submission.scene_end_ms,
    )


def _planner_evaluated_image(
    candidate_id: str,
    candidate: Candidate,
    descriptions: BrowserDescriptionOutput | None,
    *,
    reason: str,
    evaluation: CandidateEvaluationResult | None = None,
) -> PlannerEvaluatedImage:
    frame_description = (
        descriptions.description_for(candidate_id, "frame")
        if descriptions is not None
        else None
    )
    scene_description = (
        descriptions.description_for(candidate_id, "scene")
        if descriptions is not None
        else None
    )
    video_id = "_".join(candidate.id[:-1]) if len(candidate.id) > 1 else None
    visual_similarity: VisualSimilarity | None = (
        evaluation.visual_similarity
        if evaluation is not None
        and evaluation.visual_similarity != VisualSimilarity.UNKNOWN
        else VisualSimilarity.UNKNOWN
    )
    if visual_similarity == VisualSimilarity.UNKNOWN:
        visual_similarity = None
    temporal_status: TemporalStatus | None = (
        evaluation.temporal_status
        if evaluation is not None
        and evaluation.temporal_status != TemporalStatus.NOT_APPLICABLE
        else TemporalStatus.NOT_APPLICABLE
    )
    if temporal_status == TemporalStatus.NOT_APPLICABLE:
        temporal_status = None
    return PlannerEvaluatedImage(
        image_id=candidate_id,
        description=frame_description or _candidate_history_description(candidate),
        scene_description=scene_description,
        video_id=video_id,
        timestamp_ms=candidate.time_keyframe,
        reason=reason,
        visual_similarity=visual_similarity,
        temporal_status=temporal_status,
    )


def _browser_relevance_reason(
    candidate_id: str,
    descriptions: BrowserDescriptionOutput | None,
    reasons: dict[str, str],
) -> str:
    if descriptions is not None:
        if descriptions.description_for(candidate_id, "scene") is not None:
            return reasons.get(
                candidate_id,
                "The inspected scene was relevant rather than an exact match.",
            )
        if descriptions.description_for(candidate_id, "frame") is not None:
            return (
                "The browser classified this keyframe as relevant rather than an "
                "exact match."
            )
    return reasons.get(
        candidate_id,
        "The browser classified this keyframe as relevant rather than an exact match.",
    )


def _rejection_reason(
    candidate_id: str,
    rejection_result_by_id: dict[str, CandidateEvaluationResult],
    rejected_submission_by_id: dict[str, CandidateSubmission],
    unavailable_submission_by_id: dict[str, CandidateSubmission],
) -> str:
    if candidate_id in rejected_submission_by_id:
        reason = rejected_submission_by_id[candidate_id].reason
        if reason:
            return reason
    if candidate_id in unavailable_submission_by_id:
        reason = unavailable_submission_by_id[candidate_id].reason
        if reason:
            return reason
    if candidate_id in rejection_result_by_id:
        reasoning = rejection_result_by_id[candidate_id].reasoning
        if reasoning:
            return reasoning
    return "This candidate does not fully match the target."


def _acceptance_reason(evaluation: CandidateEvaluationResult) -> str:
    return evaluation.reasoning or (
        "The browser found an exact AVS scene and submission verification accepted it."
    )


def _avs_match_feedback(
    evaluations: list[CandidateEvaluationResult],
) -> str:
    accepted_count = sum(1 for evaluation in evaluations if evaluation.matches_task)
    rejected_count = len(evaluations) - accepted_count
    return (
        f"The browser/verifier path accepted {accepted_count} and rejected "
        f"{rejected_count} exact AVS scene candidates. Continue searching for new "
        "matching scenes and do not resubmit accepted candidates."
    )


def _candidate_result_summary(
    *,
    candidate_count: int,
    relevant_count: int,
) -> str:
    keyframe_noun = "keyframe" if candidate_count == 1 else "keyframes"
    relevant_verb = "was" if relevant_count == 1 else "were"
    irrelevant_count = max(candidate_count - relevant_count, 0)
    irrelevant_verb = "was" if irrelevant_count == 1 else "were"
    return (
        f"Out of {candidate_count} {keyframe_noun}, "
        f"{relevant_count} {relevant_verb} relevant and "
        f"{irrelevant_count} {irrelevant_verb} irrelevant."
    )


def _watched_segments(
    candidates_by_id: dict[str, Candidate],
    inspected_candidate_ids: set[str],
    *,
    inspected_video_intervals_ms: dict[str, tuple[int, int]] | None = None,
) -> list[PlannerWatchedSegment]:
    watched: list[PlannerWatchedSegment] = []
    intervals = inspected_video_intervals_ms or {}
    for key, candidate in candidates_by_id.items():
        if key not in inspected_candidate_ids:
            continue
        start_ms, end_ms = intervals.get(
            key,
            (candidate.time_scene_start, candidate.time_scene_end),
        )
        watched.append(
            PlannerWatchedSegment(
                candidate_id=key,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
    return watched


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
