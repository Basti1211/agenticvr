"""Internal state objects for VQA answer orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from application.types.vqa.models import VQASegment, VQASegmentAttempt
from application.types.retrieval.candidate import Candidate
from application.types.vlm.prompt import ImageInput, VideoInput


@dataclass(frozen=True)
class VQAAnswerContext:
    question: str
    visible_task_text: str | None
    task_images: list[ImageInput]
    reference_video: VideoInput | None
    source_video_id: str | None
    source_candidate_id: str | None
    temporal_mode: str | None
    time_specific: bool
    previous_answer_submissions: list[dict[str, Any]]


@dataclass(frozen=True)
class VQASceneSelection:
    query_candidates: list[Candidate]
    scene_candidates: list[Candidate]
    retrieval_duration_ms: float
    vqa_strategy: str


@dataclass
class VQASceneAnswerState:
    segments: list[VQASegment]
    attempts: list[VQASegmentAttempt]
    inspected_scene_candidates: list[Candidate]
    calls: list[dict[str, Any]]
    previous_segment_count: int = 0
    previous_inspected_scene_keys: set[tuple[str, int, int]] | None = None
    scene_limit: int | None = None
