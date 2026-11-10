"""Submission verification helpers for retrieval loop answers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from application.system.retrieval_loop.candidates import candidate_id
from application.tasks.models import TaskCreate, TaskType
from application.types.answerer.models import (
    AnswererOutput,
    AnswererValidation,
    AnswererValidationStatus,
)
from application.types.retrieval.candidate import Candidate
from application.types.retrieval_loop.models import (
    CandidateSubmission,
    CandidateSubmissionStatus,
)


_COMPETITION_YEAR_PATTERN = re.compile(r"\bvbs(?:20)?(?P<year>\d{2})(?=\D|$)", re.I)
_JUDGEMENT_ANSWERS_DIR = Path(__file__).resolve().parents[3] / "scripts"
_POINT_SUBMISSION_TASK_TYPES = {
    TaskType.AVS,
    TaskType.TEXTUAL_KIS,
    TaskType.VISUAL_KIS,
}


def verify_submission(*, task: TaskCreate, candidate: Candidate) -> CandidateSubmission:
    """Check a submitted candidate against an explicit answer target, if provided."""

    submitted_id = candidate_id(candidate)
    competition_status = _competition_judgement_status(task, candidate)
    if competition_status is not None:
        avs_submission = _avs_submission_fields(candidate, competition_status)
        if competition_status == CandidateSubmissionStatus.ACCEPTED:
            return CandidateSubmission(
                candidate_id=submitted_id,
                accepted=True,
                reason=(
                    "Submitted candidate matched an accepted competition judgement "
                    f"answer for {task.source_task_name}."
                ),
                source="competition_judgement",
                **avs_submission,
            )
        if competition_status == CandidateSubmissionStatus.REJECTED:
            return CandidateSubmission(
                candidate_id=submitted_id,
                accepted=False,
                reason=(
                    "Submitted candidate matched a rejected competition judgement "
                    f"answer for {task.source_task_name}."
                ),
                source="competition_judgement",
                **avs_submission,
            )
        return CandidateSubmission(
            candidate_id=submitted_id,
            accepted=False,
            reason=(
                "Submitted candidate was not present in the accepted or rejected "
                f"competition judgement answers for {task.source_task_name}."
            ),
            source="competition_judgement",
            **avs_submission,
        )

    submission_fields = _point_submission_fields(
        task.task_type,
        candidate,
        CandidateSubmissionStatus.ACCEPTED
        if task.task_type == TaskType.AVS and task.input.answer_video is None
        else None,
    )
    target = task.input.answer_video
    if target is None:
        return CandidateSubmission(
            candidate_id=submitted_id,
            accepted=True,
            reason="Accepted because no explicit target was available.",
            source="unverified",
            **submission_fields,
        )

    if not _candidate_matches_video_target(candidate, target.name):
        return CandidateSubmission(
            candidate_id=submitted_id,
            accepted=False,
            reason=(
                "Submitted candidate was rejected by the external target check because "
                "it is not from the expected target video."
            ),
            source="target_verifier",
            **_point_submission_fields(
                task.task_type,
                candidate,
                CandidateSubmissionStatus.REJECTED,
            ),
        )

    if not _candidate_matches_target_time(
        task.task_type,
        candidate,
        start_ms=target.start_ms,
        end_ms=target.end_ms,
    ):
        return CandidateSubmission(
            candidate_id=submitted_id,
            accepted=False,
            reason=(
                "Submitted candidate was rejected by the external target check because "
                "its timestamp is outside the expected target time range."
            ),
            source="target_verifier",
            **_point_submission_fields(
                task.task_type,
                candidate,
                CandidateSubmissionStatus.REJECTED,
            ),
        )

    return CandidateSubmission(
        candidate_id=submitted_id,
        accepted=True,
        reason="Submitted candidate matched the provided target video and time range.",
        source="target_verifier",
        **_point_submission_fields(
            task.task_type,
            candidate,
            CandidateSubmissionStatus.ACCEPTED,
        ),
    )


def verify_answerer_output(
    *, task: TaskCreate, output: AnswererOutput
) -> AnswererValidation:
    """Check an answerer guess only against explicit ground truth when available."""

    if task.input.answer_text is not None:
        return verify_text_answer(task=task, answer=output.answer)

    if task.input.answer_video is None:
        return AnswererValidation(
            status=AnswererValidationStatus.UNAVAILABLE,
            reason="Ground-truth answer video is unavailable for this task.",
        )

    candidate = Candidate(
        uri="",
        rank=None,
        score=None,
        id=[output.video_id, str(output.timestamp_ms)],
        time_keyframe=output.timestamp_ms,
        time_scene_start=output.timestamp_ms,
        time_scene_end=output.timestamp_ms,
    )
    submission = verify_submission(task=task, candidate=candidate)
    return AnswererValidation(
        status=(
            AnswererValidationStatus.ACCEPTED
            if submission.accepted
            else AnswererValidationStatus.REJECTED
        ),
        reason=submission.reason or "Ground-truth validation completed.",
        source=submission.source,
    )


def verify_text_answer(*, task: TaskCreate, answer: str) -> AnswererValidation:
    """Check a textual answer against task or competition ground truth."""

    expected = task.input.answer_text
    if expected is None:
        return _verify_competition_text_answer(task, answer)

    if _normalized_text_answer(answer) == _normalized_text_answer(expected):
        return AnswererValidation(
            status=AnswererValidationStatus.ACCEPTED,
            reason="Submitted textual answer matched the provided answer text.",
            source="answer_text_verifier",
        )

    return AnswererValidation(
        status=AnswererValidationStatus.REJECTED,
        reason=(
            "Submitted textual answer was rejected by the external answer-text "
            f"check. Expected {expected!r}, got {answer!r}."
        ),
        source="answer_text_verifier",
    )


def _verify_competition_text_answer(
    task: TaskCreate,
    answer: str,
) -> AnswererValidation:
    if task.task_type != TaskType.QA or not task.source_task_name:
        return AnswererValidation(
            status=AnswererValidationStatus.UNAVAILABLE,
            reason="Ground-truth answer text is unavailable for this task.",
        )

    task_judgements = _competition_task_judgements(task.source_task_name)
    if task_judgements is None:
        return AnswererValidation(
            status=AnswererValidationStatus.UNAVAILABLE,
            reason=(
                "Competition answer judgements are unavailable for "
                f"{task.source_task_name}."
            ),
        )

    accepted = _text_answers(task_judgements.get("accepted"))
    rejected = _text_answers(task_judgements.get("rejected"))
    if not accepted and not rejected:
        return AnswererValidation(
            status=AnswererValidationStatus.UNAVAILABLE,
            reason=(
                "Competition text-answer judgements are unavailable for "
                f"{task.source_task_name}."
            ),
        )

    normalized_answer = _normalized_text_answer(answer)
    if normalized_answer in {_normalized_text_answer(item) for item in accepted}:
        return AnswererValidation(
            status=AnswererValidationStatus.ACCEPTED,
            reason=(
                "Submitted textual answer matched an accepted competition "
                f"judgement answer for {task.source_task_name}."
            ),
            source="competition_judgement",
        )
    if normalized_answer in {_normalized_text_answer(item) for item in rejected}:
        return AnswererValidation(
            status=AnswererValidationStatus.REJECTED,
            reason=(
                "Submitted textual answer matched a rejected competition "
                f"judgement answer for {task.source_task_name}."
            ),
            source="competition_judgement",
        )
    return AnswererValidation(
        status=AnswererValidationStatus.UNAVAILABLE,
        reason=(
            "Submitted textual answer was not present in the accepted or rejected "
            f"competition judgement answers for {task.source_task_name}."
        ),
        source="competition_judgement",
    )


def _candidate_matches_video_target(candidate: Candidate, target_name: str) -> bool:
    candidate_video_id = candidate.id[0] if candidate.id else ""
    return _normalized_video_id(candidate_video_id) == _normalized_video_id(target_name)


def _candidate_overlaps_target_time(
    candidate: Candidate,
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> bool:
    if start_ms is None and end_ms is None:
        return True

    keyframe = candidate.time_keyframe
    if (start_ms is None or keyframe >= start_ms) and (
        end_ms is None or keyframe <= end_ms
    ):
        return True

    scene_start = candidate.time_scene_start
    scene_end = candidate.time_scene_end
    if start_ms is None:
        return scene_start <= end_ms if end_ms is not None else True
    if end_ms is None:
        return scene_end >= start_ms
    return scene_start <= end_ms and scene_end >= start_ms


def _candidate_matches_target_time(
    task_type: TaskType,
    candidate: Candidate,
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> bool:
    """Use point submissions for AVS/KIS and scene overlap for other tasks."""

    if task_type in _POINT_SUBMISSION_TASK_TYPES:
        return _timestamp_is_in_interval(
            candidate.time_keyframe,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    return _candidate_overlaps_target_time(
        candidate,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _timestamp_is_in_interval(
    timestamp_ms: int,
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> bool:
    return (start_ms is None or timestamp_ms >= start_ms) and (
        end_ms is None or timestamp_ms <= end_ms
    )


def _normalized_video_id(value: str) -> str:
    normalized = value.strip().strip("/")
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    if parsed.scheme in {"http", "https", "file"}:
        normalized = parsed.path
    name = Path(normalized).name or normalized
    return Path(name).stem.lower()


def _normalized_text_answer(value: str) -> str:
    return " ".join(value.strip().casefold().strip("`\"'.,;:!?").split())


def _avs_submission_fields(
    candidate: Candidate,
    status: CandidateSubmissionStatus | None,
) -> dict[str, int | str | CandidateSubmissionStatus | None]:
    video_id = "_".join(candidate.id[:-1]) if len(candidate.id) > 1 else (
        candidate.id[0] if candidate.id else None
    )
    return {
        "status": status,
        "video_id": video_id,
        "timestamp_ms": candidate.time_keyframe,
        # AVS submits one timestamp. The browser's predicted scene remains in
        # browser scene-analysis metadata for inspection and debugging.
        "scene_start_ms": None,
        "scene_end_ms": None,
    }


def _point_submission_fields(
    task_type: TaskType,
    candidate: Candidate,
    status: CandidateSubmissionStatus | None,
) -> dict[str, int | str | CandidateSubmissionStatus | None]:
    """Serialize the single timestamp submitted for AVS and KIS tasks."""

    if task_type == TaskType.AVS:
        return _avs_submission_fields(candidate, status)
    if task_type in {TaskType.TEXTUAL_KIS, TaskType.VISUAL_KIS}:
        return {"timestamp_ms": candidate.time_keyframe}
    return {}


def _competition_judgement_status(
    task: TaskCreate,
    candidate: Candidate,
) -> CandidateSubmissionStatus | None:
    if task.task_type != TaskType.AVS or not task.source_task_name:
        return None

    task_judgements = _competition_task_judgements(task.source_task_name)
    if task_judgements is None:
        return None

    accepted = _video_answers(task_judgements.get("accepted"))
    rejected = _video_answers(task_judgements.get("rejected"))
    if not accepted and not rejected:
        return None

    if _candidate_matches_any_competition_answer(candidate, accepted):
        return CandidateSubmissionStatus.ACCEPTED
    if _candidate_matches_any_competition_answer(candidate, rejected):
        return CandidateSubmissionStatus.REJECTED
    return CandidateSubmissionStatus.UNAVAILABLE


def _competition_task_judgements(source_task_name: str) -> dict[str, Any] | None:
    year = _competition_year(source_task_name)
    if year is None:
        return None
    normalized_names = _normalized_competition_task_names(source_task_name)
    for item in _competition_judgement_answers(year):
        name = item.get("name")
        if not isinstance(name, str):
            continue
        if _normalized_task_name(name) in normalized_names:
            return item
    return None


@lru_cache(maxsize=None)
def _competition_judgement_answers(year: int) -> tuple[dict[str, Any], ...]:
    path = _JUDGEMENT_ANSWERS_DIR / f"vbs{year}_judgement_answers.json"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, list):
        return ()
    return tuple(item for item in loaded if isinstance(item, dict))


def _competition_year(source_task_name: str) -> int | None:
    match = _COMPETITION_YEAR_PATTERN.search(source_task_name)
    if match is None:
        return None
    return 2000 + int(match.group("year"))


def _normalized_competition_task_names(source_task_name: str) -> set[str]:
    normalized = _normalized_task_name(source_task_name)
    names = {normalized}
    match = _COMPETITION_YEAR_PATTERN.search(normalized)
    if match is not None:
        short_year = match.group("year")
        names.add(
            normalized[: match.start()]
            + f"vbs20{short_year}"
            + normalized[match.end() :]
        )
        names.add(
            normalized[: match.start()] + f"vbs{short_year}" + normalized[match.end() :]
        )
    return names


def _normalized_task_name(value: str) -> str:
    return value.strip().casefold()


def _video_answers(raw_answers: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_answers, list):
        return []
    return [
        answer
        for answer in raw_answers
        if isinstance(answer, dict)
        and answer.get("file") is not None
        and answer.get("start_ms") is not None
        and answer.get("end_ms") is not None
    ]


def _text_answers(raw_answers: Any) -> list[str]:
    if not isinstance(raw_answers, list):
        return []
    return [
        answer for answer in raw_answers if isinstance(answer, str) and answer.strip()
    ]


def _candidate_matches_any_competition_answer(
    candidate: Candidate,
    answers: list[dict[str, Any]],
) -> bool:
    candidate_video_id = _normalized_video_id(
        "_".join(candidate.id[:-1]) if len(candidate.id) > 1 else (
            candidate.id[0] if candidate.id else ""
        )
    )
    if not candidate_video_id:
        return False

    for answer in answers:
        if candidate_video_id != _normalized_video_id(str(answer["file"])):
            continue
        start_ms = _optional_int(answer.get("start_ms"))
        end_ms = _optional_int(answer.get("end_ms"))
        if start_ms is None or end_ms is None:
            continue
        start_ms, end_ms = min(start_ms, end_ms), max(start_ms, end_ms)
        if _timestamp_is_in_interval(
            candidate.time_keyframe,
            start_ms=start_ms,
            end_ms=end_ms,
        ):
            return True
    return False


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
