"""Data contracts for independent second-stage candidate verification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from application.types.vlm.prompt import VideoInput


class SecondStageEvaluationRequest(BaseModel):
    """The two inputs required for second-stage verification."""

    candidate_keyframe_id: str = Field(min_length=1)
    task_input_text: str = Field(min_length=1)

    @field_validator("candidate_keyframe_id", "task_input_text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Second-stage evaluator inputs must not be blank.")
        return stripped


class SecondStageEvidenceItem(BaseModel):
    """One atomic target predicate and its entity/time-bound visual evidence."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(min_length=1)
    predicate_type: Literal[
        "subject_presence",
        "entity_binding",
        "exact_action",
        "object_category",
        "attribute",
        "count",
        "spatial_relationship",
        "temporal_relationship",
        "setting",
        "other",
    ]
    evidence: str = Field(min_length=1)
    entity_binding: str = Field(
        min_length=1,
        description=(
            "Stable visible entity labels and the predicates bound to them; "
            "use 'not_applicable' only when the predicate has no entity."
        ),
    )
    temporal_binding: str = Field(
        min_length=1,
        description="Approximate clip time or interval inspected for this predicate.",
    )


class SecondStageVideoClip(BaseModel):
    """Materialized keyframe-centred video evidence."""

    video: VideoInput
    keyframe_timestamp_ms: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    fps: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> "SecondStageVideoClip":
        if self.end_ms <= self.start_ms:
            raise ValueError("Second-stage clip end must be greater than start.")
        if not self.start_ms <= self.keyframe_timestamp_ms <= self.end_ms:
            raise ValueError("Second-stage clip must contain the candidate keyframe.")
        return self


class SecondStageEvaluationResult(BaseModel):
    """Strict scene-level verdict with explicit support and failure evidence."""

    candidate_keyframe_id: str = Field(min_length=1)
    scene_description: str = Field(min_length=1)
    matching_details: list[SecondStageEvidenceItem] = Field(default_factory=list)
    non_matching_details: list[SecondStageEvidenceItem] = Field(default_factory=list)
    uncertain_details: list[SecondStageEvidenceItem] = Field(default_factory=list)
    fully_matches: bool
    reason: str = Field(min_length=1)
    clip_start_ms: int = Field(ge=0)
    clip_end_ms: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_verdict_evidence(self) -> "SecondStageEvaluationResult":
        if self.clip_end_ms <= self.clip_start_ms:
            raise ValueError("Second-stage result clip end must be greater than start.")
        if self.fully_matches:
            if not self.matching_details:
                raise ValueError("Full matches require explicit matching evidence.")
            if self.non_matching_details or self.uncertain_details:
                raise ValueError(
                    "Full matches cannot contain non-matching or uncertain details."
                )
        elif not (self.non_matching_details or self.uncertain_details):
            raise ValueError(
                "Rejected second-stage results require a non-match or uncertainty."
            )
        return self
