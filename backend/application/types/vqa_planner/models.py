"""Contracts for the video-locked QA planning agent."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from application.types.answerer.models import AnswererValidation


class VQAPlannerQueryKeyframes(BaseModel):
    action: Literal["query_keyframes"]
    query: str = Field(min_length=1)


class VQAPlannerInspectScene(BaseModel):
    action: Literal["inspect_scene"]
    keyframe_id: str = Field(min_length=1)


class VQAPlannerInspectScenes(BaseModel):
    action: Literal["inspect_scenes"]
    keyframe_ids: list[Annotated[str, Field(min_length=1)]] = Field(min_length=2)


class VQAPlannerInspectAllScenes(BaseModel):
    action: Literal["inspect_all_scenes"]
    start_timestamp_ms: Literal["START"] | Annotated[int, Field(ge=0)] = "START"
    end_timestamp_ms: Literal["END"] | Annotated[int, Field(ge=0)] = "END"

    @model_validator(mode="after")
    def validate_timestamp_range(self) -> VQAPlannerInspectAllScenes:
        if (
            isinstance(self.start_timestamp_ms, int)
            and isinstance(self.end_timestamp_ms, int)
            and self.start_timestamp_ms >= self.end_timestamp_ms
        ):
            raise ValueError("start_timestamp_ms must be before end_timestamp_ms")
        return self


class VQAPlannerScanFromStart(BaseModel):
    action: Literal["scan_from_start"]


class VQAPlannerAnswer(BaseModel):
    action: Literal["answer"]
    answer: str = Field(min_length=1)


class VQAPlannerEndWithoutAnswer(BaseModel):
    action: Literal["end_without_answer"]
    reason: str = Field(min_length=1)


VQAPlannerAction: TypeAlias = Annotated[
    VQAPlannerQueryKeyframes
    | VQAPlannerInspectScene
    | VQAPlannerInspectScenes
    | VQAPlannerInspectAllScenes
    | VQAPlannerScanFromStart
    | VQAPlannerAnswer
    | VQAPlannerEndWithoutAnswer,
    Field(discriminator="action"),
]


class VQAPlannerHistoryEntry(BaseModel):
    """One action and its compact, question-relevant result."""

    action: dict[str, Any]
    result: dict[str, Any]


class VQAPlannerRequest(BaseModel):
    """Question, matched-scene context, and private VQA-planner history."""

    question: str = Field(min_length=1)
    invocation_timestamp_ms: int = Field(ge=0)
    matched_scene_description: str | None = Field(default=None, min_length=1)
    history: list[VQAPlannerHistoryEntry] = Field(default_factory=list)


class VQAPlannerDecision(BaseModel):
    action: VQAPlannerAction
    metadata: dict[str, Any] = Field(default_factory=dict)


class VQAPlannerStep(BaseModel):
    index: int = Field(ge=1)
    decision: VQAPlannerDecision
    result: dict[str, Any]
    validation: AnswererValidation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VQAPlannerSessionResult(BaseModel):
    answer: str | None = Field(default=None, min_length=1)
    ended_without_answer: bool = False
    reason: str | None = Field(default=None, min_length=1)
    steps: list[VQAPlannerStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
