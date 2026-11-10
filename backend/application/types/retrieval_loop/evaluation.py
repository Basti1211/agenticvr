"""Candidate-evaluation records produced by retrieval-loop decisions."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

from application.types.evidence import TemporalStatus, VisualSimilarity


class CandidateEvaluationDecision(StrEnum):
    """Decision labels for candidate evidence."""

    ACCEPT = "accept"
    REJECT = "reject"


class CandidateEvaluationResult(BaseModel):
    """Browser- or system-derived decision for one candidate."""

    candidate_id: str = Field(
        min_length=1,
        description="Identifier of the evaluated candidate.",
    )
    matches_task: bool = Field(
        description="Whether the candidate satisfies the task.",
    )
    decision: CandidateEvaluationDecision | None = Field(
        default=None,
        description="Final decision for the candidate evidence.",
    )
    reasoning: str | None = Field(
        default=None,
        description="Evidence-based explanation of the decision.",
    )
    visual_similarity: VisualSimilarity = Field(
        default=VisualSimilarity.UNKNOWN,
        description="Appearance similarity assessed separately from acceptance.",
    )
    temporal_status: TemporalStatus = Field(
        default=TemporalStatus.NOT_APPLICABLE,
        description="Status of any temporal evidence used for the decision.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-essential trace metadata.",
    )

    @model_validator(mode="after")
    def validate_reasoning(self) -> "CandidateEvaluationResult":
        if self.decision is None:
            self.decision = (
                CandidateEvaluationDecision.ACCEPT
                if self.matches_task
                else CandidateEvaluationDecision.REJECT
            )
        if self.matches_task and self.decision != CandidateEvaluationDecision.ACCEPT:
            raise ValueError("Matching evaluations must use the accept decision")
        if not self.matches_task and self.decision == CandidateEvaluationDecision.ACCEPT:
            raise ValueError("Accepted evaluations must match the task")
        if (
            not self.matches_task
            and self.decision == CandidateEvaluationDecision.REJECT
            and not (self.reasoning or "").strip()
        ):
            raise ValueError("Rejected evaluations require reasoning")
        return self

    @computed_field
    @property
    def should_submit(self) -> bool:
        """Whether this candidate can be submitted as an answer."""

        return self.matches_task
