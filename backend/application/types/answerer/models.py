"""Data contracts for the independent per-iteration answerer."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

from application.tasks.models import Dataset, EncodedTask, TaskType
from application.types.vlm.prompt import VideoInput


class AnswererObservedSequence(BaseModel):
    """One deduplicated video sequence and its accumulated assessments."""

    video_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    judgement: str | None = Field(
        default=None,
        description="Strongest browser judgement for the sequence.",
    )
    observations: list[str] = Field(
        default_factory=list,
        description="Deduplicated textual browser observations.",
    )


class AnswererObservedKeyframe(BaseModel):
    """One independently represented keyframe and its accumulated evidence."""

    keyframe_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    scene_start_ms: int | None = Field(default=None, ge=0)
    scene_end_ms: int | None = Field(default=None, ge=0)
    judgement: str | None = None
    frame_description: str | None = Field(default=None, min_length=1)
    scene_description: str | None = Field(default=None, min_length=1)
    observations: list[str] = Field(default_factory=list)


class AnswererQAFinding(BaseModel):
    """One textual QA finding tied to observed video evidence."""

    video_id: str = Field(min_length=1)
    timestamp_ms: int | None = Field(default=None, ge=0)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class AnswererRejectedCandidate(BaseModel):
    """An anonymized retrieval-loop submission rejected as the exact target."""

    video_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    scene_start_ms: int | None = Field(default=None, ge=0)
    scene_end_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_optional_scene_interval(self) -> "AnswererRejectedCandidate":
        if (self.scene_start_ms is None) != (self.scene_end_ms is None):
            raise ValueError(
                "Rejected answerer evidence requires both scene interval bounds or "
                "neither."
            )
        if (
            self.scene_start_ms is not None
            and self.scene_end_ms is not None
            and self.scene_end_ms <= self.scene_start_ms
        ):
            raise ValueError(
                "Rejected answerer evidence scene_end_ms must be greater than "
                "scene_start_ms."
            )
        return self


class AnswererHistory(BaseModel):
    """Cumulative evidence summary supplied to the independent answerer."""

    observed_keyframes: list[AnswererObservedKeyframe] = Field(default_factory=list)
    qa_findings: list[AnswererQAFinding] = Field(default_factory=list)
    rejected_candidates: list[AnswererRejectedCandidate] = Field(
        default_factory=list,
        description=(
            "Anonymized candidates rejected by retrieval-loop submission checks. "
            "This excludes the independent answerer's own guesses and validations."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_observed_sequences(cls, value: Any) -> Any:
        """Accept older persisted answerer histories without regrouping new data."""

        if not isinstance(value, dict) or "observed_sequences" not in value:
            return value
        migrated = dict(value)
        sequences = migrated.pop("observed_sequences") or []
        migrated.setdefault(
            "observed_keyframes",
            [
                {
                    "keyframe_id": f"{item['video_id']}_{item['timestamp_ms']}",
                    "video_id": item["video_id"],
                    "timestamp_ms": item["timestamp_ms"],
                    "scene_start_ms": item.get("start_ms"),
                    "scene_end_ms": item.get("end_ms"),
                    "judgement": item.get("judgement"),
                    "observations": item.get("observations", []),
                }
                for raw in sequences
                for item in [
                    raw.model_dump(mode="python") if isinstance(raw, BaseModel) else raw
                ]
            ],
        )
        return migrated


class AnswererRequest(BaseModel):
    """Task and evidence supplied to the external answerer."""

    encoded_task: EncodedTask
    history: AnswererHistory = Field(default_factory=AnswererHistory)
    dataset: Dataset
    task_type: TaskType
    task_videos: list[VideoInput] = Field(
        default_factory=list,
        description="Task/reference videos allowed as answerer target context.",
    )


class AnswererOutput(BaseModel):
    """One best-guess video and timestamp emitted by the answerer."""

    video_id: str = Field(min_length=1)
    timestamp_ms: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def answer(self) -> str:
        """Return the canonical answer text shown in logs and the frontend."""

        return f"{self.video_id}, {self.timestamp_ms}"


class AnswererValidationStatus(StrEnum):
    """Ground-truth validation state for an answerer output."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class AnswererValidation(BaseModel):
    """External ground-truth validation of an answerer output."""

    status: AnswererValidationStatus
    reason: str = Field(min_length=1)
    source: str | None = Field(default=None, min_length=1)
