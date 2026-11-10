"""Question-answering component data contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from application.tasks.models import Dataset, TaskType
from application.types.vlm.prompt import ImageInput, VideoInput


class QARequest(BaseModel):
    """One visual question grounded in exactly one image or video."""

    question: str = Field(
        min_length=1,
        description="Textual question to answer from the supplied visual evidence.",
    )
    image: ImageInput | None = Field(
        default=None,
        description="Image or keyframe used as the visual evidence.",
    )
    video: VideoInput | None = Field(
        default=None,
        description="Video segment used as the temporal visual evidence.",
    )
    dataset: Dataset | None = Field(
        default=None,
        description="Optional dataset context used to tune the QA prompt.",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Optional task context used to tune the QA prompt.",
    )

    @model_validator(mode="after")
    def _require_exactly_one_media_source(self) -> "QARequest":
        if (self.image is None) == (self.video is None):
            raise ValueError("QARequest requires exactly one image or video.")
        return self


class QAOutput(BaseModel):
    """Text answer generated from a QA request."""

    answer: str = Field(
        min_length=1,
        description="Answer grounded in the supplied visual evidence.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-essential execution metadata, including VLM calls.",
    )
