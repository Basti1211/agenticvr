from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from application.tasks.models import Dataset, TaskType


class LlmAgent(StrEnum):
    """Concrete LLM-backed agents with independently configurable prompts."""

    TASK_ENCODER_IMAGE_DESCRIBER = "task_encoder_image_describer"
    TASK_ENCODER_TASK_VIDEO_DESCRIBER = "task_encoder_task_video_describer"
    TASK_ENCODER_VIDEO_SEGMENT_DESCRIBER = "task_encoder_video_segment_describer"
    TASK_ENCODER_QA_TEMPORAL_MODE_CLASSIFIER = (
        "task_encoder_qa_temporal_mode_classifier"
    )
    PLANNER = "planner"
    ANSWERER = "answerer"
    BROWSER_CLASSIFIER = "browser_classifier"
    QA = "qa"
    VQA = "vqa"
    VQA_PLANNER = "vqa_planner"
    SECOND_STAGE_EVALUATOR = "second_stage_evaluator"


class ImageInput(BaseModel):
    allowed_extensions: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".webp"}

    identifier: str = Field(description="Internal image identifier for result mapping.")
    file: Path = Field(description="Path to an image file.")

    @field_validator("file")
    @classmethod
    def validate_image_file(cls, value: Path) -> Path:
        if value.suffix.lower() not in cls.allowed_extensions:
            raise ValueError(
                f"Image file must use one of: {', '.join(sorted(cls.allowed_extensions))}"
            )
        return value


class VideoInput(BaseModel):
    allowed_extensions: ClassVar[set[str]] = {".mp4", ".mov", ".m4v", ".webm"}

    identifier: str = Field(description="Internal video identifier for result mapping.")
    file: Path = Field(description="Path to a video file.")

    @field_validator("file")
    @classmethod
    def validate_video_file(cls, value: Path) -> Path:
        if value.suffix.lower() not in cls.allowed_extensions:
            raise ValueError(
                f"Video file must use one of: {', '.join(sorted(cls.allowed_extensions))}"
            )
        return value


class PromptContext(BaseModel):
    llm_agent: LlmAgent = Field(description="LLM-backed module receiving the prompt.")
    dataset: Dataset | None = Field(
        default=None,
        description="Dataset used for prompt selection and call tracing.",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Task type used for prompt selection and call tracing.",
    )
    video_sampling_fps: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional per-call video sampling rate overriding the agent default."
        ),
    )


class Prompt(BaseModel):
    system_prompt: str = Field(description="System prompt for the VLM.")
    context_prompt: str = Field(description="Context prompt for the VLM.")
    temperature: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional request-specific sampling temperature. When omitted, the "
            "configured model temperature is used."
        ),
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of tokens the VLM should generate.",
    )
    task_text: str | None = Field(
        default=None,
        description="Textual task-target evidence rendered in the shared task context.",
    )
    task_images: list[ImageInput] = Field(
        default_factory=list,
        description=(
            "Temporally ordered images that define the task target, distinct from "
            "candidate/context images."
        ),
    )
    task_videos: list[VideoInput] = Field(
        default_factory=list,
        description=(
            "Videos that define the task target, distinct from candidate/context videos."
        ),
    )
    context_images: list[ImageInput] = Field(
        description="Candidate or supporting images included in the context."
    )
    context_videos: list[VideoInput] = Field(
        description="Videos included in the context."
    )
    context: PromptContext | None = Field(
        default=None,
        description="Optional task, dataset, and concrete-agent call metadata.",
    )


class BatchedPrompt(BaseModel):
    prompts: list[Prompt] = Field(
        min_length=1,
        description="Prompts to process as a batch.",
    )
