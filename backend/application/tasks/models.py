"""Pydantic models for task setup and stored task representations."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class TaskType(StrEnum):
    """Supported VBS/LSC task families."""

    VISUAL_KIS = "v-kis"
    TEXTUAL_KIS = "t-kis"
    QA = "qa"
    AVS = "avs"


class Dataset(StrEnum):
    """Supported task datasets."""

    MVK = "mvk"
    V3C = "v3c"
    LHE = "lhe"


class TaskStatus(StrEnum):
    """Lifecycle state for a stored task representation."""

    CREATED = "created"
    ENCODE_PENDING = "encode_pending"
    ENCODED = "encoded"
    FAILED = "failed"


class VideoSegment(BaseModel):
    """A named video segment in milliseconds."""

    name: str = Field(min_length=1, description="Video identifier or file name.")
    start_ms: int | None = Field(
        default=None,
        ge=0,
        description="Optional inclusive segment start in milliseconds.",
    )
    end_ms: int | None = Field(
        default=None,
        ge=0,
        description="Optional exclusive segment end in milliseconds.",
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> "VideoSegment":
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms <= self.start_ms
        ):
            raise ValueError("end_ms must be greater than start_ms")
        return self


class MediaInput(BaseModel):
    """Metadata for user-provided media used by a task."""

    name: str = Field(min_length=1, description="Public media label or media id.")
    media_type: Literal["image", "video"] = Field(description="Submitted media type.")
    mime_type: str | None = Field(
        default=None, description="Browser supplied MIME type."
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Browser supplied file size.",
    )
    local_path: str | None = Field(
        default=None,
        description="Backend-local file path for uploaded media.",
    )


class TaskInput(BaseModel):
    """Concrete input fields needed to define a task."""

    description: str | None = Field(
        default=None,
        description="Textual task description, query, or question.",
    )
    media: MediaInput | None = Field(
        default=None,
        description="Uploaded image/video metadata for visual tasks.",
    )
    task_video: VideoSegment | None = Field(
        default=None,
        description="Known task video segment, if the task references one.",
    )
    answer_text: str | None = Field(
        default=None,
        description="Optional known/expected textual answer.",
    )
    answer_video: VideoSegment | None = Field(
        default=None,
        description="Optional known/expected video answer segment.",
    )


class TaskCreate(BaseModel):
    """Payload accepted when a user or script submits a task."""

    task_type: TaskType = Field(description="Task family to execute.")
    dataset: Dataset = Field(description="Dataset where the task should run.")
    input: TaskInput = Field(description="Concrete task input.")
    source_task_name: str | None = Field(
        default=None,
        description="Optional imported competition task name, such as vbs24-kis-t1.",
    )

    @model_validator(mode="after")
    def validate_task_specific_inputs(self) -> "TaskCreate":
        description = (self.input.description or "").strip()

        if self.task_type == TaskType.VISUAL_KIS and not (
            self.input.media or self.input.task_video
        ):
            raise ValueError("v-kis tasks require uploaded media or a task video")

        if self.task_type in {TaskType.TEXTUAL_KIS, TaskType.QA, TaskType.AVS}:
            if not description:
                raise ValueError(
                    f"{self.task_type.value} tasks require description text"
                )

        return self


class EncodedTask(BaseModel):
    """Output boundary for the future task encoder."""

    status: Literal["pending", "encoded"] = Field(
        default="pending",
        description="Encoder status. Pending until a reliable encoder exists.",
    )
    text: str | None = Field(
        default=None,
        description="Encoded textual representation when available.",
    )
    media_refs: list[str] = Field(
        default_factory=list,
        description="Encoded or referenced media identifiers.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Implementation notes from the encoder.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-essential trace metadata, including VLM calls.",
    )


class TaskRepresentation(BaseModel):
    """Stored task representation used by the backend pipeline."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: TaskStatus = Field(default=TaskStatus.CREATED)
    task_type: TaskType
    dataset: Dataset
    input: TaskInput
    source_task_name: str | None = None
    encoded_task: EncodedTask | None = None
    task_input_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Effective task description visible to each configured loop module."
        ),
    )


class TaskOption(BaseModel):
    """Frontend/script metadata for a task type."""

    value: TaskType
    label: str
    required_inputs: list[Literal["text", "media"]]
    description: str


class DatasetOption(BaseModel):
    """Frontend/script metadata for a dataset."""

    value: Dataset
    label: str
    description: str


class TaskOptionsResponse(BaseModel):
    """Available task and dataset options."""

    task_types: list[TaskOption]
    datasets: list[DatasetOption]


class ImportedTaskMedia(BaseModel):
    """Previewable media referenced by an imported task."""

    media_type: Literal["image", "video"]
    file: str
    url: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None


class ImportedTaskOption(BaseModel):
    """Normalized imported VBS task metadata for the frontend."""

    name: str
    task_type: TaskType
    dataset: Dataset
    input_text: str | None = None
    input_text_hints: list[str] = Field(default_factory=list)
    input_media: ImportedTaskMedia | None = None
    target_video: VideoSegment | None = None


class TaskRunLogSummary(BaseModel):
    """Summary metadata for one imported task execution log."""

    log_id: str
    run_id: str
    setting_name: str | None = None
    task_name: str
    task_type: TaskType
    dataset: Dataset
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    source_file: str | None = None
    max_iterations: int
    max_candidates: int
    hint_index: int | Literal["iterative"] | None = None
    hide_avs_submission_ratings_from_planner: bool = False
    iteration_count: int = 0
    final_candidate_id: str | None = None
    final_answer: str | None = None
    duration_ms: float | None = None
    error: str | None = None


class TaskRunLog(TaskRunLogSummary):
    """Full imported task execution log."""

    task_id: str | None = None
    target_video: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    task_input_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Effective encoded task description visible to each loop module."
        ),
    )
    final_submission: dict[str, Any] | None = None
    final_answerer_output: dict[str, Any] | None = None
    final_answerer_validation: dict[str, Any] | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    traceback: str | None = None
