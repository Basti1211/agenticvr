"""Browser component data contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from application.tasks.models import Dataset, TaskType
from application.types.vlm.prompt import ImageInput, VideoInput


class BrowserJudgement(StrEnum):
    """Candidate relevance labels produced by the browser component."""

    MATCH = "match"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class KeyframeDecision(StrEnum):
    """Whether a retrieved keyframe justifies inspecting its surrounding scene."""

    INSPECT_SCENE = "inspect_scene"
    DISCARD = "discard"


class SceneVerdict(StrEnum):
    """Authoritative browser decision after watching a candidate scene."""

    EXACT = "exact"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class BrowserMediaCandidate(BaseModel):
    """One media candidate returned by the data layer."""

    image: ImageInput | None = Field(
        default=None,
        description="Image/keyframe returned by the data layer for VLM inspection.",
    )
    images: list[ImageInput] = Field(
        default_factory=list,
        description="Ordered keyframes inspected together as one candidate.",
    )
    video: VideoInput | None = Field(
        default=None,
        description="Video/segment returned by the data layer for VLM inspection.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured data-layer metadata. Not sent to the VLM.",
    )

    @model_validator(mode="after")
    def validate_candidate_evidence(self) -> "BrowserMediaCandidate":
        if self.image is None and not self.images and self.video is None:
            raise ValueError("Browser candidate requires image or video evidence")
        evidence_kinds = sum(
            [
                self.image is not None,
                bool(self.images),
                self.video is not None,
            ]
        )
        if evidence_kinds > 1:
            raise ValueError("Browser candidate supports one evidence kind at a time")
        return self

    @computed_field
    @property
    def identifier(self) -> str:
        """Stable candidate id used for internal browser result mapping."""

        if self.image is not None:
            return self.image.identifier
        if self.images:
            candidate_id = self.metadata.get("id")
            if isinstance(candidate_id, list) and candidate_id:
                return "_".join(str(part) for part in candidate_id)
            return self.images[0].identifier
        evidence = self.video
        if evidence is None:
            raise ValueError("Browser candidate requires image or video evidence")
        return evidence.identifier


BrowserCandidate = BrowserMediaCandidate


class BrowserRequest(BaseModel):
    """Input for browser inspection after a retrieval step."""

    task_description: str = Field(
        description="Condensed description from the task encoder.",
    )
    candidates: list[BrowserMediaCandidate] = Field(
        min_length=1,
        description="Media candidates returned by the data layer.",
    )
    task_images: list[ImageInput] = Field(
        default_factory=list,
        description="Anonymous task reference frames supplied alongside the task description.",
    )
    task_videos: list[VideoInput] = Field(
        default_factory=list,
        description="Anonymous task/reference videos supplied as target context.",
    )
    task_type: TaskType | None = Field(
        default=None,
        description="Optional task type used to tune browser strictness.",
    )
    dataset: Dataset | None = Field(
        default=None,
        description="Optional dataset used to tune browser strictness.",
    )

    @model_validator(mode="after")
    def validate_unique_candidate_ids(self) -> "BrowserRequest":
        seen: set[str] = set()
        duplicates: set[str] = set()

        for candidate in self.candidates:
            if candidate.identifier in seen:
                duplicates.add(candidate.identifier)
            seen.add(candidate.identifier)

        if duplicates:
            raise ValueError(
                "Browser candidate identifiers must be unique: "
                f"{', '.join(sorted(duplicates))}"
            )
        return self


class BrowserFrameAnalysis(BaseModel):
    """Static keyframe routing evidence produced in one browser call."""

    candidate_id: str = Field(min_length=1)
    frame_description: str = Field(min_length=1)
    decision: KeyframeDecision
    possible_target_moment: str | None = None
    visible_support: list[str] = Field(default_factory=list)
    visible_conflicts: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class BrowserFrameAnalysisOutput(BaseModel):
    """Frame analyses for one retrieved candidate page."""

    results: list[BrowserFrameAnalysis] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrowserSceneAnalysis(BaseModel):
    """Authoritative evidence produced after watching a candidate scene."""

    candidate_id: str = Field(min_length=1)
    scene_description: str = Field(min_length=1)
    verdict: SceneVerdict
    matching_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    best_match_start_seconds: float | None = Field(default=None, ge=0)
    best_match_end_seconds: float | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_match_interval(self) -> "BrowserSceneAnalysis":
        start = self.best_match_start_seconds
        end = self.best_match_end_seconds
        if self.verdict == SceneVerdict.EXACT and (start is None or end is None):
            raise ValueError("Exact scene verdicts require a best-match interval")
        if (start is None) != (end is None):
            raise ValueError("Best-match start and end must be supplied together")
        if start is not None and end is not None and end <= start:
            raise ValueError("Best-match end must be greater than its start")
        return self


class BrowserSceneAnalysisOutput(BaseModel):
    """Scene analyses for visually strong candidates."""

    results: list[BrowserSceneAnalysis] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BrowserClassification(BaseModel):
    """Browser classification for one media candidate."""

    candidate_id: str = Field(
        min_length=1,
        description="Identifier of the inspected candidate.",
    )
    judgement: BrowserJudgement = Field(
        description="Whether the candidate is a match, relevant, or irrelevant.",
    )
    reasoning: str | None = Field(
        default=None,
        description=(
            "Reason for the final keyframe or inspected-scene decision."
        ),
    )
    matching_evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class BrowserClassificationOutput(BaseModel):
    """Per-candidate browser decisions for a retrieval result page."""

    results: list[BrowserClassification] = Field(
        min_length=1,
        description="Per-candidate classifications derived from browser analyses.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-essential trace metadata, including VLM calls.",
    )

    @computed_field
    @property
    def match_candidate_ids(self) -> list[str]:
        """Candidate ids that the browser classified as complete matches."""

        return [
            result.candidate_id
            for result in self.results
            if result.judgement == BrowserJudgement.MATCH
        ]

    @computed_field
    @property
    def exact_candidate_ids(self) -> list[str]:
        """Candidate ids whose surrounding scenes are exact browser matches."""

        return self.match_candidate_ids

    @computed_field
    @property
    def relevant_candidate_ids(self) -> list[str]:
        """Candidate ids that can be used for browser reasoning/search feedback."""

        return [
            result.candidate_id
            for result in self.results
            if result.judgement == BrowserJudgement.RELEVANT
        ]

    @computed_field
    @property
    def irrelevant_candidate_ids(self) -> list[str]:
        """Candidate ids summarized together for irrelevant feedback."""

        return [
            result.candidate_id
            for result in self.results
            if result.judgement == BrowserJudgement.IRRELEVANT
        ]

    @computed_field
    @property
    def has_match(self) -> bool:
        """Whether the browser classified at least one complete match."""

        return bool(self.match_candidate_ids)

    @computed_field
    @property
    def has_exact_candidate(self) -> bool:
        """Whether at least one inspected scene is an exact browser match."""

        return bool(self.exact_candidate_ids)


class BrowserReasoning(BaseModel):
    """Planner feedback synthesized from the browser analyses."""

    candidate_ids: list[str] = Field(
        min_length=1,
        description="Candidate ids covered by this reasoning.",
    )
    judgement: BrowserJudgement = Field(
        description="Classification this reasoning explains.",
    )
    reasoning: str = Field(
        min_length=1,
        description="Short reason retained from the applicable browser analysis.",
    )


class BrowserReasoningOutput(BaseModel):
    """Planner-facing feedback for relevant and irrelevant classifications."""

    relevant: list[BrowserReasoning] = Field(default_factory=list)
    irrelevant: BrowserReasoning | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-essential trace metadata, including VLM calls.",
    )


class BrowserDescription(BaseModel):
    """A factual description of one keyframe or inspected scene."""

    candidate_id: str = Field(
        min_length=1,
        description="Identifier of the keyframe represented by this evidence.",
    )
    media_kind: Literal["frame", "scene"] = Field(
        description="Whether the description covers a keyframe or a scene clip.",
    )
    description: str = Field(
        min_length=1,
        description="Short factual description grounded in the supplied media.",
    )


class BrowserDescriptionOutput(BaseModel):
    """Frame and scene descriptions extracted from browser analyses."""

    results: list[BrowserDescription] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-essential trace metadata, including VLM calls.",
    )

    def description_for(self, candidate_id: str, media_kind: str) -> str | None:
        """Return the description for one candidate/media pair, when present."""

        return next(
            (
                result.description
                for result in self.results
                if result.candidate_id == candidate_id
                and result.media_kind == media_kind
            ),
            None,
        )
