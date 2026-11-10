"""Mutable execution state and explicit outcomes for the retrieval loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from application.system.timing import elapsed_ms
from application.tasks.models import EncodedTask, TaskCreate
from application.types.planner.models import PlannerCondensedHistory, PlannerDecision
from application.types.retrieval.candidate import Candidate
from application.types.retrieval_loop.models import (
    AVSMatch,
    CandidateSubmission,
    RetrievalHistoryEntry,
    RetrievalLoopIteration,
)
from application.types.retrieval_loop.evaluation import CandidateEvaluationResult


@dataclass
class RunContext:
    """State shared by every stage of one retrieval-loop invocation."""

    task: TaskCreate
    validation_task: TaskCreate
    encoded: EncodedTask
    timings_ms: dict[str, float]
    iterations: list[RetrievalLoopIteration]
    full_history: list[RetrievalHistoryEntry]
    condensed_history: PlannerCondensedHistory
    description_hints: tuple[str, ...] = ()
    description_hint_started_at: float | None = None
    description_hint_elapsed_offset_seconds: float = 0.0
    description_hint_index: int | None = None
    irrelevant_candidate_ids: set[str] = field(default_factory=set)
    deferred_match_candidates: dict[str, Candidate] = field(default_factory=dict)
    planner_reference_candidates: list[Candidate] = field(default_factory=list)
    avs_matches_by_id: dict[str, AVSMatch] = field(default_factory=dict)
    seen_candidate_ids: set[str] = field(default_factory=set)
    vqa_searched_video_ids: set[str] = field(default_factory=set)
    vqa_answer_feedback: list[dict[str, Any]] = field(default_factory=list)
    rejected_vqa_answers: list[str] = field(default_factory=list)
    submitted_vqa_answers: set[str] = field(default_factory=set)
    submitted_candidates: list[CandidateSubmission] = field(default_factory=list)
    on_iteration_progress: Any | None = None


@dataclass
class IterationContext:
    """State shared by stages belonging to one planned iteration."""

    run: RunContext
    decision: PlannerDecision
    iteration: RetrievalLoopIteration
    started_at: float

    @property
    def index(self) -> int:
        return self.iteration.index

    def finish(self) -> None:
        """Record the iteration duration before optional answerer work."""

        self.iteration.timings_ms["total"] = elapsed_ms(self.started_at)


@dataclass(frozen=True)
class IterationOutcome:
    """The control-flow result of a completed iteration."""

    iteration: RetrievalLoopIteration
    final_candidate: Candidate | None = None
    final_answer: str | None = None
    final_evaluation: CandidateEvaluationResult | None = None
    final_submission: CandidateSubmission | None = None
    avs_matches: tuple[AVSMatch, ...] = ()

    @property
    def found(self) -> bool:
        return self.final_answer is not None or bool(
            self.final_submission and self.final_submission.accepted
        )
