"""Data contracts for the automated retrieval loop."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field

from application.tasks.models import EncodedTask
from application.types.answerer.models import AnswererOutput, AnswererValidation
from application.types.browser.models import (
    BrowserClassificationOutput,
    BrowserDescriptionOutput,
    BrowserReasoningOutput,
)
from application.types.planner.models import PlannerCondensedHistory
from application.types.qa.models import QAOutput
from application.types.retrieval.candidate import Candidate
from application.types.retrieval_loop.evaluation import CandidateEvaluationResult


class RetrievalLoopStatus(StrEnum):
    """Terminal state of one retrieval loop run."""

    FOUND = "found"
    NOT_FOUND = "not_found"
    ENCODE_PENDING = "encode_pending"
    TIMEOUT = "timeout"


class CandidateSubmissionStatus(StrEnum):
    """External submission classification."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class CandidateSubmission(BaseModel):
    """An answer that the loop attempted to submit for final verification."""

    candidate_id: str = Field(
        min_length=1,
        description=(
            "Submitted candidate identifier, or a stable synthetic id for text "
            "answer submissions."
        ),
    )
    answer: str | None = Field(
        default=None,
        min_length=1,
        description="Submitted textual answer, used by QA tasks.",
    )
    accepted: bool = Field(
        description="Whether the submitted answer passed final verification.",
    )
    status: CandidateSubmissionStatus | None = Field(
        default=None,
        description=(
            "External classification of the submitted candidate. `unavailable` "
            "means no competition judgement or explicit ground truth classified it. "
            "AVS and QA text submissions populate this field."
        ),
    )
    reason: str | None = Field(
        default=None,
        description="Reason the submitted answer was accepted, rejected, or unavailable.",
    )
    source: str = Field(
        default="unverified",
        min_length=1,
        description="Component that produced the final submission decision.",
    )
    video_id: str | None = Field(
        default=None,
        min_length=1,
        description="Submitted source video id, when known.",
    )
    timestamp_ms: int | None = Field(
        default=None,
        ge=0,
        description="Submitted keyframe timestamp in milliseconds, when known.",
    )
    scene_start_ms: int | None = Field(
        default=None,
        ge=0,
        description="Start of the submitted candidate scene, when known.",
    )
    scene_end_ms: int | None = Field(
        default=None,
        ge=0,
        description="End of the submitted candidate scene, when known.",
    )


class AVSMatch(BaseModel):
    """One browser-exact AVS candidate accepted by submission verification."""

    candidate: Candidate = Field(description="Accepted AVS candidate evidence.")
    evaluation: CandidateEvaluationResult = Field(
        description=(
            "Compatibility record containing the authoritative browser decision."
        )
    )


class RetrievalHistoryEntry(BaseModel):
    """One full-fidelity event from the retrieval loop execution trace."""

    stage: str = Field(min_length=1, description="Component or operation name.")
    kind: str = Field(
        min_length=1,
        description="Event kind, such as llm, retrieval, or module.",
    )
    request: Any | None = Field(default=None, description="Request/query payload.")
    response: Any | None = Field(default=None, description="Response/answer payload.")
    error: str | None = Field(default=None, description="Error, if the event failed.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalLoopIteration(BaseModel):
    """One planner/retrieval/browser/verification cycle."""

    index: int = Field(ge=1, description="One-based loop iteration index.")
    action: str = Field(min_length=1, description="Planner action.")
    planner_purpose: str | None = Field(
        default=None,
        min_length=1,
        description="Planner-stated reason for choosing this action.",
    )
    planner_policy_rule: str | None = Field(
        default=None,
        min_length=1,
        description="Action-selection policy rule cited by the planner.",
    )
    endpoint: str = Field(
        min_length=1,
        description="Data-layer endpoint or internal module used.",
    )
    retrieved_candidate_ids: list[str] | None = Field(
        default=None,
        description=(
            "Ordered candidate ids returned by the query before loop-level filtering. "
            "None means that this iteration did not complete a retrieval call."
        ),
    )
    candidates: list[Candidate] = Field(
        default_factory=list,
        description="New candidates passed to the browser.",
    )
    filtered_candidates: list[Candidate] = Field(
        default_factory=list,
        description=(
            "Previously seen candidates returned by the data layer but withheld "
            "from browser processing."
        ),
    )
    retrieval_error: str | None = Field(
        default=None,
        description=(
            "Retryable retrieval/parser/data-layer error produced by this action."
        ),
    )
    browser_output: BrowserClassificationOutput | None = Field(
        default=None,
        description="Browser classifications for retrieved candidates.",
    )
    browser_description_output: BrowserDescriptionOutput | None = Field(
        default=None,
        description=(
            "Per-keyframe frame descriptions and optional inspected-scene "
            "descriptions produced by the browser."
        ),
    )
    browser_reasoning_output: BrowserReasoningOutput | None = Field(
        default=None,
        description="Browser reasoning used to build condensed planner feedback.",
    )
    qa_output: QAOutput | None = Field(
        default=None,
        description="Text answer returned when this iteration executes IQA or VQA.",
    )
    evaluator_results: list[CandidateEvaluationResult] = Field(
        default_factory=list,
        description=(
            "Records for browser-exact decisions and QA candidate state."
        ),
    )
    submissions: list[CandidateSubmission] = Field(
        default_factory=list,
        description=(
            "Candidates submitted for final verification and their result. Direct "
            "answers can be submitted without candidate evaluation."
        ),
    )
    answerer_output: AnswererOutput | None = Field(
        default=None,
        description="Independent best-guess answer produced after this iteration.",
    )
    answerer_validation: AnswererValidation | None = Field(
        default=None,
        description="External ground-truth validation of the answerer output.",
    )
    answerer_error: str | None = Field(
        default=None,
        description=(
            "Answerer failure recorded without affecting retrieval-loop control flow."
        ),
    )
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Runtime timings for this iteration, in milliseconds.",
    )

    @computed_field
    @property
    def candidate_ids(self) -> list[str]:
        """Stable string ids for candidates returned in this iteration."""

        return [_candidate_id(candidate) for candidate in self.candidates]

    @computed_field
    @property
    def filtered_candidate_ids(self) -> list[str]:
        """Stable string ids for candidates hidden from active processing."""

        return [_candidate_id(candidate) for candidate in self.filtered_candidates]


class RetrievalLoopResult(BaseModel):
    """Output of the basic automated retrieval loop."""

    status: RetrievalLoopStatus = Field(description="Final loop status.")
    encoded_task: EncodedTask = Field(description="Encoded task used by the planner.")
    iterations: list[RetrievalLoopIteration] = Field(default_factory=list)
    full_history: list[RetrievalHistoryEntry] = Field(
        default_factory=list,
        description="Full execution trace: LLM calls, retrieval calls, and module IO.",
    )
    condensed_planner_history: PlannerCondensedHistory = Field(
        default_factory=PlannerCondensedHistory,
        description="Compact feedback history passed to the planner LLM.",
    )
    final_candidate: Candidate | None = Field(
        default=None,
        description="Candidate accepted as the final answer, if found.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Textual final answer, used by QA tasks.",
    )
    final_evaluation: CandidateEvaluationResult | None = Field(
        default=None,
        description=(
            "Legacy-compatible browser-exact or VQA decision that ended the loop. "
            "This is absent for direct answers."
        ),
    )
    final_submission: CandidateSubmission | None = Field(
        default=None,
        description="Submission result that ended the loop, if found.",
    )
    avs_matches: list[AVSMatch] = Field(
        default_factory=list,
        description=("Ordered unique candidates accepted across all AVS iterations."),
    )
    timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Top-level runtime timings for the loop, in milliseconds.",
    )

    @computed_field
    @property
    def final_candidate_id(self) -> str | None:
        """Accepted candidate id, if any."""

        return _candidate_id(self.final_candidate) if self.final_candidate else None

    @computed_field
    @property
    def avs_candidate_ids(self) -> list[str]:
        """Stable ids for all accepted AVS candidates."""

        return [_candidate_id(match.candidate) for match in self.avs_matches]


def _candidate_id(candidate: Candidate) -> str:
    return "_".join(candidate.id)
