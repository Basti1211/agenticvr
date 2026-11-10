"""Map full retrieval traces to the answerer's evidence-only history."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from application.types.answerer.models import (
    AnswererHistory,
    AnswererObservedKeyframe,
    AnswererQAFinding,
    AnswererRejectedCandidate,
)
from application.types.retrieval_loop.models import RetrievalHistoryEntry


_JUDGEMENT_PRIORITY = {
    "irrelevant": 1,
    "relevant": 2,
    "match": 3,
    "rejected": 4,
    "accepted": 5,
}
_VIDEO_INTERVAL_PATTERN = re.compile(r"^(.*):(\d+)-(\d+)$")


def full_history_to_answerer_history(
    full_history: list[RetrievalHistoryEntry],
) -> AnswererHistory:
    """Keep one independent evidence record per retrieved keyframe."""

    keyframes: dict[str, AnswererObservedKeyframe] = {}
    qa_entries: list[RetrievalHistoryEntry] = []
    rejected_candidates: dict[tuple[str, int], AnswererRejectedCandidate] = {}

    for entry in full_history:
        if entry.kind == "llm":
            continue
        if entry.stage == "retrieval":
            for payload in _as_list(entry.response):
                _record_candidate(payload, keyframes)
        elif entry.stage == "direct_answer.materialize":
            _record_candidate(entry.response, keyframes)
        elif entry.stage == "browser.classify":
            for result in _as_list(_as_dict(entry.response).get("results")):
                payload = _as_dict(result)
                _record_assessment(
                    candidate_id=payload.get("candidate_id"),
                    judgement=payload.get("judgement"),
                    reasoning=payload.get("reasoning"),
                    keyframes=keyframes,
                )
        elif entry.stage == "browser.frame_analysis":
            for result in _as_list(_as_dict(entry.response).get("results")):
                payload = _as_dict(result)
                _record_description(
                    candidate_id=payload.get("candidate_id"),
                    media_kind="frame",
                    description=payload.get("frame_description"),
                    keyframes=keyframes,
                )
        elif entry.stage == "browser.scene_analysis":
            for result in _as_list(_as_dict(entry.response).get("results")):
                payload = _as_dict(result)
                _record_description(
                    candidate_id=payload.get("candidate_id"),
                    media_kind="scene",
                    description=payload.get("scene_description"),
                    keyframes=keyframes,
                )
        elif entry.stage == "qa":
            qa_entries.append(entry)
        elif entry.stage == "submission_verifier":
            _record_submission_rejection(
                entry,
                keyframes=keyframes,
                rejected_candidates=rejected_candidates,
            )

    qa_findings = [
        finding
        for entry in qa_entries
        if (finding := _qa_finding(entry, keyframes)) is not None
    ]
    return AnswererHistory(
        observed_keyframes=sorted(
            keyframes.values(),
            key=lambda item: (item.video_id, item.timestamp_ms, item.keyframe_id),
        ),
        qa_findings=sorted(
            _unique_qa_findings(qa_findings),
            key=lambda item: (
                item.video_id,
                item.start_ms if item.start_ms is not None else -1,
                item.timestamp_ms if item.timestamp_ms is not None else -1,
                item.question,
            ),
        ),
        rejected_candidates=sorted(
            rejected_candidates.values(),
            key=lambda item: (item.video_id, item.timestamp_ms),
        ),
    )


def _record_submission_rejection(
    entry: RetrievalHistoryEntry,
    *,
    keyframes: dict[str, AnswererObservedKeyframe],
    rejected_candidates: dict[tuple[str, int], AnswererRejectedCandidate],
) -> None:
    """Expose safe retrieval-loop rejection facts without answerer feedback loops."""

    response = _as_dict(entry.response)
    if response.get("accepted") is not False or response.get("status") == "unavailable":
        return
    request = _as_dict(entry.request)
    candidate_id = response.get("candidate_id") or request.get("candidate_id")
    if not isinstance(candidate_id, str):
        return
    keyframe = keyframes.get(candidate_id)
    if keyframe is None:
        return

    key = (keyframe.video_id, keyframe.timestamp_ms)
    scene_interval = _matching_observed_scene(keyframe, keyframes.values())
    rejected_candidates[key] = AnswererRejectedCandidate(
        video_id=keyframe.video_id,
        timestamp_ms=keyframe.timestamp_ms,
        scene_start_ms=scene_interval[0] if scene_interval is not None else None,
        scene_end_ms=scene_interval[1] if scene_interval is not None else None,
    )
    _record_assessment(
        candidate_id=candidate_id,
        judgement="rejected",
        reasoning=(
            "The retrieval loop's external submission checker rejected this "
            "candidate as the exact target."
        ),
        keyframes=keyframes,
    )


def _matching_observed_scene(
    rejected: AnswererObservedKeyframe,
    keyframes: Any,
) -> tuple[int, int] | None:
    """Find the narrowest known scene containing a rejected answer timestamp."""

    matching: list[tuple[int, int]] = []
    for keyframe in keyframes:
        if keyframe.video_id != rejected.video_id:
            continue
        start_ms = keyframe.scene_start_ms
        end_ms = keyframe.scene_end_ms
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            continue
        if start_ms <= rejected.timestamp_ms <= end_ms:
            matching.append((start_ms, end_ms))
    if not matching:
        return None
    return min(matching, key=lambda interval: interval[1] - interval[0])


def _record_candidate(
    value: Any,
    keyframes: dict[str, AnswererObservedKeyframe],
) -> None:
    payload = _as_dict(value)
    id_parts = payload.get("id")
    if not isinstance(id_parts, list) or not id_parts:
        return
    identifiers = [str(part).strip() for part in id_parts]
    if any(not part for part in identifiers):
        return
    timestamp_ms = _non_negative_int(payload.get("time_keyframe"))
    start_ms = _non_negative_int(payload.get("time_scene_start"))
    end_ms = _non_negative_int(payload.get("time_scene_end"))
    if timestamp_ms is None:
        return

    candidate_id = "_".join(identifiers)
    video_id = "_".join(identifiers[:-1]) if len(identifiers) > 1 else identifiers[0]
    keyframes.setdefault(
        candidate_id,
        AnswererObservedKeyframe(
            keyframe_id=candidate_id,
            video_id=video_id,
            timestamp_ms=timestamp_ms,
            scene_start_ms=start_ms,
            scene_end_ms=end_ms,
        ),
    )


def _record_assessment(
    *,
    candidate_id: Any,
    judgement: Any,
    reasoning: Any,
    keyframes: dict[str, AnswererObservedKeyframe],
) -> None:
    if not isinstance(candidate_id, str) or not isinstance(judgement, str):
        return
    keyframe = keyframes.get(candidate_id)
    if keyframe is None:
        return
    normalized_judgement = judgement.strip().lower()
    previous_priority = _JUDGEMENT_PRIORITY.get(keyframe.judgement or "", 0)
    next_priority = _JUDGEMENT_PRIORITY.get(normalized_judgement, 0)
    if next_priority > previous_priority:
        keyframe.judgement = normalized_judgement
    if isinstance(reasoning, str):
        observation = reasoning.strip()
        if observation and observation not in keyframe.observations:
            keyframe.observations.append(observation)


def _record_description(
    *,
    candidate_id: Any,
    media_kind: Any,
    description: Any,
    keyframes: dict[str, AnswererObservedKeyframe],
) -> None:
    if not all(
        isinstance(value, str) for value in (candidate_id, media_kind, description)
    ):
        return
    keyframe = keyframes.get(candidate_id)
    if keyframe is None:
        return
    normalized = description.strip()
    if not normalized:
        return
    if media_kind == "frame":
        keyframe.frame_description = normalized
    elif media_kind == "scene":
        keyframe.scene_description = normalized


def _qa_finding(
    entry: RetrievalHistoryEntry,
    keyframes: dict[str, AnswererObservedKeyframe],
) -> AnswererQAFinding | None:
    request = _as_dict(entry.request)
    response = _as_dict(entry.response)
    question = request.get("question")
    answer = response.get("answer")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(answer, str) or not answer.strip():
        return None

    image = _as_dict(request.get("image"))
    image_id = image.get("identifier")
    if isinstance(image_id, str) and image_id in keyframes:
        keyframe = keyframes[image_id]
        return AnswererQAFinding(
            video_id=keyframe.video_id,
            timestamp_ms=keyframe.timestamp_ms,
            question=question.strip(),
            answer=answer.strip(),
        )

    video = _as_dict(request.get("video"))
    identifier = video.get("identifier")
    if isinstance(identifier, str):
        match = _VIDEO_INTERVAL_PATTERN.fullmatch(identifier)
        if match is not None and match.group(1):
            return AnswererQAFinding(
                video_id=match.group(1),
                start_ms=int(match.group(2)),
                end_ms=int(match.group(3)),
                question=question.strip(),
                answer=answer.strip(),
            )
    return None


def _unique_qa_findings(
    findings: list[AnswererQAFinding],
) -> list[AnswererQAFinding]:
    unique: list[AnswererQAFinding] = []
    seen: set[tuple[Any, ...]] = set()
    for finding in findings:
        key = (
            finding.video_id,
            finding.timestamp_ms,
            finding.start_ms,
            finding.end_ms,
            finding.question,
            finding.answer,
        )
        if key not in seen:
            unique.append(finding)
            seen.add(key)
    return unique


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _non_negative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
