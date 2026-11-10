"""Planner component data contracts."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field, computed_field, model_validator

from application.tasks.models import EncodedTask, TaskCreate
from application.types.retrieval.queries import Query
from application.types.vlm.prompt import ImageInput, VideoInput
from application.types.evidence import TemporalStatus, VisualSimilarity


class PlannerActionSpec(BaseModel):
    """One retrieval action available to the planner."""

    code: str = Field(
        min_length=1,
        description="Short action token used at the start of planner output.",
    )
    name: str = Field(min_length=1, description="Human-readable action name.")
    syntax: str = Field(
        min_length=1,
        description="Exact output syntax expected by the retrieval parser.",
    )
    description: str = Field(
        min_length=1,
        description="What the action does and when it is useful.",
    )


def default_available_actions() -> list[PlannerActionSpec]:
    """Return the retrieval actions currently understood by the data layer parser."""

    return [
        PlannerActionSpec(
            code="nl",
            name="Text Query",
            syntax="nl <text query> [--one-result-per-video] [--unseen]",
            description="Search for keyframes by natural-language textual description. Returns keyframes that best match the specified moment. Add --one-result-per-video to return at most one keyframe per video and --unseen to skip keyframes already displayed in this task.",
        ),
        PlannerActionSpec(
            code="lnl",
            name="Localized Text Query",
            syntax="lnl <text query> <coordinates> [--one-result-per-video] [--unseen]",
            description="Search using a natural-language description in the selected image region with 4 coordinates <0,1> (for example '[0.0, 0.0, 0.5, 0.2]'). Add --one-result-per-video to return at most one keyframe per video and --unseen to skip keyframes already displayed in this task.",
        ),
        PlannerActionSpec(
            code="t",
            name="Temporal Query",
            syntax="t <first text> || <second text> [--one-result-per-video] [--unseen]",
            description="Search for two separate events or states by natural-language textual description. Useful when searching for temporal dependencies. Add --one-result-per-video to return at most one keyframe per video and --unseen to skip keyframes already displayed in this task.",
        ),
        PlannerActionSpec(
            code="im",
            name="Image Query",
            syntax="im <image_id> [--one-result-per-video] [--unseen]",
            description="Search visually similar keyframes using one image id. Add --one-result-per-video to return at most one keyframe per video and --unseen to skip keyframes already displayed in this task.",
        ),
        PlannerActionSpec(
            code="b",
            name="Bayes Update",
            syntax="b <image_id> [image_id ...] [--one-result-per-video] [--unseen]",
            description="Refine the last search result using image ids that are similar to the intended target. Returns an updated, relevance-ranked list of keyframes. Add --one-result-per-video to return at most one keyframe per video and --unseen to skip keyframes already displayed in this task.",
        ),
        PlannerActionSpec(
            code="lvs",
            name="Load Video Summary",
            syntax="lvs <image_id> [--unseen]",
            description="Loads keyframes of the video surrounding the specified keyframe id. Add --unseen to skip keyframes already displayed in this task.",
        ),
        PlannerActionSpec(
            code="iv",
            name="Intra Video Query",
            syntax="iv <video_id> <text query> [--unseen]",
            description=(
                "Search within one specific video using textual description. Returns "
                "the best-ranked keyframes matching the text query from the specified "
                "video. IMPORTANT: iv may be executed at most once per video per task. "
                "After iv has searched a video, do not use iv for that video again, even "
                "with different query text or --unseen. Before choosing iv, check "
                "iv_searched_video_ids in the prior search evidence; every listed video "
                "id is forbidden for iv. Add --unseen to skip keyframes already "
                "displayed in this task."
            ),
        ),
        PlannerActionSpec(
            code="lm",
            name="Load More",
            syntax="lm [--one-result-per-video] [--unseen]",
            description="Loads more candidates from the previous retrieval query. Useful when the initial query contained a lot of promising results. Add --one-result-per-video to continue with at most one keyframe per video and --unseen to skip keyframes already displayed in this task.",
        ),
        PlannerActionSpec(
            code="back",
            name="Back",
            syntax="back <steps>",
            description="Move backward through data-layer action history and restore the candidates returned after that previous action. If steps exceed available history, the data layer restores the first action result.",
        ),
        PlannerActionSpec(
            code="IQA",
            name="Image Question Answering",
            syntax="IQA <keyframe_id> || <question>",
            description=(
                "Ask a focused question about one keyframe id from prior search "
                "evidence. The textual answer is returned as planner evidence."
            ),
        ),
        PlannerActionSpec(
            code="VQA",
            name="Video Question Answering",
            syntax="VQA <video_name> <start_ms> <end_ms> || <question>",
            description=(
                "Ask a focused question about the occurence or absence of a detail in the specified video interval. The textual answer is returned for your next step as evidence."
            ),
        ),
        PlannerActionSpec(
            code="AN",
            name="Direct Answer",
            syntax="AN <video_id> <timestamp_ms>",
            description=(
                "Submit one exact frame directly for final evaluation when prior "
                "evidence is sufficient. The video id must appear in prior evidence."
            ),
        ),
    ]


class PlannerEvaluatedImage(BaseModel):
    """One evaluator-described image candidate; contains ids and text only."""

    image_id: str = Field(
        min_length=1,
        description="Image/keyframe id returned by the retrieval layer or evaluator.",
    )
    description: str = Field(
        min_length=1,
        description="Evaluator description of what is visible in the image.",
    )
    reason: str = Field(
        min_length=1,
        description="Evaluator explanation for why this image belongs in this category.",
    )
    scene_description: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Description of the inspected scene clip, when this keyframe triggered "
            "temporal browser inspection."
        ),
    )
    video_id: str | None = Field(default=None, min_length=1)
    timestamp_ms: int | None = Field(default=None, ge=0)
    visual_similarity: VisualSimilarity | None = None
    temporal_status: TemporalStatus | None = None


class PlannerIrrelevantImageGroup(BaseModel):
    """A grouped summary for images that are not useful for the task."""

    image_ids: list[str] = Field(
        default_factory=list,
        description="Image/keyframe ids covered by this irrelevant group.",
    )


class PlannerEvaluatorResult(BaseModel):
    """Browser or system feedback from one retrieval round.

    The planner receives only image ids and evaluator-written text. It must not
    expect raw images, videos, screenshots, or browser media in history.
    """

    relevant_images: list[PlannerEvaluatedImage] = Field(
        default_factory=list,
        description=(
            "Images useful for the task or refinement. For each image, the "
            "description says what is visible and reason explains why it fits "
            "the encoded task description."
        ),
    )
    highly_relevant_images: list[PlannerEvaluatedImage] = Field(
        default_factory=list,
        description=(
            "Images that look close or were considered possible matches, "
            "but are not final matches. For each image, the description says "
            "what is visible and reason explains why it is not a match."
        ),
    )
    irrelevant_groups: list[PlannerIrrelevantImageGroup] = Field(
        default_factory=list,
        description=(
            "Grouped summaries for irrelevant images. Use these as negative "
            "evidence without listing a separate explanation for every image."
        ),
    )
    evaluator_feedback: str | None = Field(
        default=None,
        description=(
            "Optional overall evaluator or system guidance for the next round, "
            "including retrieval-parser or data-layer errors."
        ),
    )
    irrelevant_reason: str | None = Field(
        default=None,
        description="Shared reason why the irrelevant result group was not useful.",
    )

    @property
    def not_match_images(self) -> list[PlannerEvaluatedImage]:
        """Backward-compatible alias for rejected near matches."""

        return self.highly_relevant_images


class ImageQAAction(BaseModel):
    """Planner request to ask a question about one known keyframe."""

    keyframe_id: str = Field(
        min_length=1,
        description="Keyframe id that appears in prior planner evidence.",
    )
    question: str = Field(
        min_length=1,
        description="Question to answer from the selected keyframe.",
    )


class VideoQAAction(BaseModel):
    """Planner request to ask a question about one known video interval."""

    video_name: str = Field(
        min_length=1,
        description="Video identifier that appears in prior planner evidence.",
    )
    start_ms: int = Field(
        ge=0,
        description="Inclusive start of the video evidence in milliseconds.",
    )
    end_ms: int = Field(
        ge=0,
        description="Exclusive end of the video evidence in milliseconds.",
    )
    question: str = Field(
        min_length=1,
        description="Question to answer from the selected video interval.",
    )

    @model_validator(mode="after")
    def _require_ordered_interval(self) -> "VideoQAAction":
        if self.end_ms <= self.start_ms:
            raise ValueError("VQA end_ms must be greater than start_ms.")
        return self


PlannerOperation: TypeAlias = Query | ImageQAAction | VideoQAAction


class PlannerQAResult(BaseModel):
    """Question and answer evidence returned to the planner."""

    media_id: str = Field(
        min_length=1,
        description="Keyframe or video identifier inspected by QA.",
    )
    start_ms: int | None = Field(
        default=None,
        ge=0,
        description="Video interval start for VQA; absent for IQA.",
    )
    end_ms: int | None = Field(
        default=None,
        ge=0,
        description="Video interval end for VQA; absent for IQA.",
    )
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_optional_interval(self) -> "PlannerQAResult":
        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError("QA history requires both start_ms and end_ms or neither.")
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms <= self.start_ms
        ):
            raise ValueError("QA history end_ms must be greater than start_ms.")
        return self


class PlannerWatchedSegment(BaseModel):
    """Scene interval that the browser watched for one retrieved candidate."""

    candidate_id: str = Field(
        min_length=1,
        description="Candidate whose scene was presented to the browser.",
    )
    start_ms: int = Field(
        ge=0,
        description="Inclusive start of the watched scene in milliseconds.",
    )
    end_ms: int = Field(
        ge=0,
        description="Exclusive end of the watched scene in milliseconds.",
    )

    @model_validator(mode="after")
    def _require_ordered_interval(self) -> "PlannerWatchedSegment":
        if self.end_ms <= self.start_ms:
            raise ValueError(
                "Browser history segment end_ms must be greater than start_ms."
            )
        return self


class PlannerWronglySubmittedCandidate(BaseModel):
    """A video frame and its known local scene rejected by the external checker."""

    video_id: str = Field(
        min_length=1,
        description="Video identifier that was submitted as an answer.",
    )
    timestamp_ms: int = Field(
        ge=0,
        description="Exact submitted timestamp in milliseconds.",
    )
    scene_start_ms: int | None = Field(
        default=None,
        ge=0,
        description="Start of the rejected inspected scene, when known.",
    )
    scene_end_ms: int | None = Field(
        default=None,
        ge=0,
        description="End of the rejected inspected scene, when known.",
    )

    @model_validator(mode="after")
    def _validate_optional_scene_interval(self) -> "PlannerWronglySubmittedCandidate":
        if (self.scene_start_ms is None) != (self.scene_end_ms is None):
            raise ValueError(
                "Rejected planner answers require both scene_start_ms and "
                "scene_end_ms or neither."
            )
        if (
            self.scene_start_ms is not None
            and self.scene_end_ms is not None
            and self.scene_end_ms <= self.scene_start_ms
        ):
            raise ValueError(
                "Rejected planner answer scene_end_ms must be greater than "
                "scene_start_ms."
            )
        return self


class PlannerSubmittedCandidate(BaseModel):
    """A video frame or scene submitted to the external checker."""

    video_id: str = Field(
        min_length=1,
        description="Video identifier that was submitted as an answer.",
    )
    timestamp_ms: int = Field(
        ge=0,
        description="Submitted keyframe timestamp in milliseconds.",
    )
    status: Literal["accepted", "rejected", "unavailable"] | None = Field(
        default=None,
        description=(
            "External submission status. `unavailable` means the submission was "
            "not present in the available competition judgement answers. None means "
            "the rating was deliberately withheld from the planner."
        ),
    )
    candidate_id: str | None = Field(
        default=None,
        min_length=1,
        description="Submitted keyframe candidate id, when known.",
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


class PlannerHistoryStep(BaseModel):
    """One previous action and the feedback it produced."""

    action: str = Field(
        min_length=1,
        description="Retrieval action that was executed.",
    )
    purpose: str | None = Field(
        default=None,
        min_length=1,
        description="Planner-stated reason for choosing this action.",
    )
    policy_rule: str | None = Field(
        default=None,
        min_length=1,
        description="Action-selection policy rule cited by the planner.",
    )
    action_valid: bool | None = Field(
        default=None,
        description=(
            "Whether the retrieval layer accepted this action. False actions do not "
            "replace the most recent valid action for transition-rule checks. None "
            "supports history created before this field was introduced."
        ),
    )
    result_candidate_ids: list[str] | None = Field(
        default=None,
        description=(
            "Candidate ids returned by the data layer for this action, in result order."
        ),
    )
    result_summary: str | None = Field(
        default=None,
        description=(
            "Compact browser/evaluator summary of what the returned candidates showed."
        ),
    )
    watched_segments: list[PlannerWatchedSegment] | None = Field(
        default=None,
        description=(
            "Scene intervals presented to the browser for the candidates in this step."
        ),
    )
    evaluator_result: PlannerEvaluatorResult = Field(
        default_factory=PlannerEvaluatorResult,
        description=(
            "Browser or system result for this action, containing image "
            "ids and textual explanations only; no raw images or videos."
        ),
    )
    qa_result: PlannerQAResult | None = Field(
        default=None,
        description=(
            "Structured textual evidence returned by an IQA or VQA action, when "
            "this history step represents question answering."
        ),
    )
    wrongly_submitted_candidates: list[PlannerWronglySubmittedCandidate] = Field(
        default_factory=list,
        description=(
            "Exact video/timestamp answers rejected by an external submission "
            "checker during this step."
        ),
    )
    submitted_candidates: list[PlannerSubmittedCandidate] = Field(
        default_factory=list,
        description=(
            "Video/timestamp answers submitted during this step, including accepted, "
            "rejected, and unavailable competition-judgement classifications."
        ),
    )


class PlannerResultGroups(BaseModel):
    """Candidate ids grouped by browser/evaluator judgement for one action."""

    relevant: list[str] = Field(
        default_factory=list,
        description="Candidate ids classified as relevant refinement evidence.",
    )
    irrelevant: list[str] = Field(
        default_factory=list,
        description="Candidate ids classified as irrelevant for the target.",
    )
    highly_relevant: list[str] = Field(
        default_factory=list,
        description=(
            "Candidate ids that were close enough to inspect as possible matches, "
            "but were rejected or still need refinement."
        ),
    )


class PlannerActionHistory(BaseModel):
    """One valid planner action used for the task, without duplicated evidence."""

    action: str = Field(
        min_length=1,
        description="Valid planner action that was executed.",
    )
    purpose: str | None = Field(
        default=None,
        min_length=1,
        description="Planner-stated reason for choosing this action.",
    )
    policy_rule: str | None = Field(
        default=None,
        min_length=1,
        description="Action-selection policy rule cited by the planner.",
    )
    result_candidate_ids: list[str] | None = Field(
        default=None,
        exclude=True,
        description=(
            "Relevant and highly-relevant candidate ids retained from the data-layer "
            "results, in result order. Irrelevant ids are retained separately in "
            "result_groups.irrelevant."
        ),
    )
    result_groups: PlannerResultGroups = Field(
        default_factory=PlannerResultGroups,
        description="Returned candidate ids grouped by relevance judgement.",
    )
    result_summary: str | None = Field(
        default=None,
        description="Compact summary of the result set.",
    )
    watched_segments: list[PlannerWatchedSegment] | None = Field(
        default=None,
        description=(
            "Scene intervals presented to the browser for retained relevant or "
            "highly-relevant keyframes."
        ),
    )
    evaluator_feedback: str | None = Field(
        default=None,
        description="Optional non-image guidance produced for this action.",
    )
    irrelevant_reason: str | None = Field(
        default=None,
        description=(
            "Shared reason why the irrelevant result group was not useful. This "
            "does not include irrelevant image descriptions."
        ),
    )
    qa_result: PlannerQAResult | None = Field(
        default=None,
        description="Textual IQA/VQA answer produced by this action.",
    )


class PlannerInvalidAttempt(BaseModel):
    """Invalid planner output or rejected retrieval action after the latest valid action."""

    action: str = Field(
        min_length=1,
        description="Invalid planner output or retrieval action.",
    )
    purpose: str | None = Field(default=None, min_length=1)
    policy_rule: str | None = Field(default=None, min_length=1)
    result_summary: str | None = Field(
        default=None,
        description="Compact description of why the attempt failed.",
    )
    evaluator_feedback: str | None = Field(
        default=None,
        description="Guidance for recovering from the invalid attempt.",
    )


class PlannerCanonicalEvidence(BaseModel):
    """One deduplicated scene-level description/reason for useful evidence."""

    category: Literal["relevant", "highly_relevant"] = Field(
        description="Highest usefulness category assigned to this scene.",
    )
    keyframe_ids: list[str] = Field(
        default_factory=list,
        description="Keyframe ids from this scene that appeared in result groups.",
    )
    video_id: str | None = Field(
        default=None,
        description="Video id for this scene, when it can be inferred.",
    )
    start_ms: int | None = Field(
        default=None,
        ge=0,
        description="Inclusive watched scene start in milliseconds.",
    )
    end_ms: int | None = Field(
        default=None,
        ge=0,
        description="Exclusive watched scene end in milliseconds.",
    )
    description: str = Field(
        min_length=1,
        description="Scene-level visual description.",
    )
    reason: str = Field(
        min_length=1,
        description="Why this scene is relevant or highly relevant evidence.",
    )

    @model_validator(mode="after")
    def _validate_optional_interval(self) -> "PlannerCanonicalEvidence":
        if (self.start_ms is None) != (self.end_ms is None):
            raise ValueError(
                "Canonical evidence requires both start_ms and end_ms or neither."
            )
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms <= self.start_ms
        ):
            raise ValueError("Canonical evidence end_ms must be greater than start_ms.")
        return self


class PlannerKeyframeEvidence(BaseModel):
    """One canonical history entry for one retrieved keyframe."""

    category: Literal["relevant", "highly_relevant"]
    keyframe_id: str = Field(min_length=1)
    video_id: str | None = Field(default=None, min_length=1)
    timestamp_ms: int | None = Field(default=None, ge=0)
    frame_description: str = Field(min_length=1)
    scene_description: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    inspected_segment: PlannerWatchedSegment | None = None
    visual_similarity: VisualSimilarity | None = None
    temporal_status: TemporalStatus | None = None


class PlannerCondensedHistory(BaseModel):
    """Structured compact planner history passed to the planner LLM."""

    latest_valid_action: PlannerActionHistory | None = Field(
        default=None,
        description="Most recent valid planner action that was actually used.",
    )
    used_actions: list[PlannerActionHistory] = Field(
        default_factory=list,
        description=(
            "Valid planner actions used for this task before latest_valid_action."
        ),
    )
    invalid_attempts_after_latest_action: list[PlannerInvalidAttempt] = Field(
        default_factory=list,
        description=(
            "Invalid planner outputs or retrieval attempts since the latest valid "
            "used action."
        ),
    )
    keyframe_evidence: list[PlannerKeyframeEvidence] = Field(
        default_factory=list,
        description=(
            "Descriptions and reasoning deduplicated strictly by keyframe id. Scene "
            "descriptions are retained only when a clip was actually inspected."
        ),
    )
    vqa_searched_video_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Source videos already searched by final VQA without producing an "
            "accepted text answer. The planner should avoid searching these videos "
            "again for QA."
        ),
    )
    rejected_vqa_answers: list[str] = Field(
        default_factory=list,
        description=(
            "Text answers submitted for QA and rejected by the outside answer "
            "checker. Full VLM inputs remain only in full history."
        ),
    )
    wrongly_submitted_candidates: list[PlannerWronglySubmittedCandidate] = Field(
        default_factory=list,
        description=(
            "Deduplicated video/timestamp answers rejected by the external "
            "submission checker, with rejected local scene intervals when known. "
            "The planner must not submit these exact answers again and should avoid "
            "other timestamps in the same rejected scene."
        ),
    )
    submitted_candidates: list[PlannerSubmittedCandidate] = Field(
        default_factory=list,
        description=(
            "Deduplicated video/timestamp/scene answers already submitted. The "
            "planner should search for distinct new images and avoid submitting "
            "the same frame or overlapping scene again."
        ),
    )

    @model_validator(mode="after")
    def _compact_action_result_details(self) -> "PlannerCondensedHistory":
        """Keep compact action links while discarding bulky duplicated details."""

        for action in [*self.used_actions, self.latest_valid_action]:
            if action is not None:
                _compact_action_results(action)
        return self

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_steps(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, cls):
            return value
        if isinstance(value, list):
            return cls.from_steps(value).model_dump(mode="python")
        if isinstance(value, dict) and "canonical_evidence" in value:
            migrated = dict(value)
            legacy_items = migrated.pop("canonical_evidence") or []
            if not migrated.get("keyframe_evidence"):
                keyframe_evidence = []
                for item in legacy_items:
                    payload = (
                        item.model_dump(mode="python")
                        if isinstance(item, BaseModel)
                        else dict(item)
                    )
                    for keyframe_id in payload.get("keyframe_ids", []):
                        segment = None
                        if (
                            payload.get("start_ms") is not None
                            and payload.get("end_ms") is not None
                        ):
                            segment = {
                                "candidate_id": keyframe_id,
                                "start_ms": payload["start_ms"],
                                "end_ms": payload["end_ms"],
                            }
                        keyframe_evidence.append(
                            {
                                "category": payload["category"],
                                "keyframe_id": keyframe_id,
                                "video_id": payload.get("video_id"),
                                "timestamp_ms": None,
                                "frame_description": payload["description"],
                                "scene_description": payload["description"]
                                if segment
                                else None,
                                "reason": payload["reason"],
                                "inspected_segment": segment,
                            }
                        )
                migrated["keyframe_evidence"] = keyframe_evidence
            return migrated
        return value

    @classmethod
    def from_steps(
        cls, steps: list[PlannerHistoryStep | dict[str, Any]]
    ) -> "PlannerCondensedHistory":
        history = cls()
        for raw_step in steps:
            step = (
                raw_step
                if isinstance(raw_step, PlannerHistoryStep)
                else PlannerHistoryStep.model_validate(raw_step)
            )
            history.add_step(step)
        return history

    def add_step(self, step: PlannerHistoryStep) -> None:
        """Merge one legacy action step into the structured condensed history."""

        if step.action_valid is False or _is_legacy_invalid_history_step(step):
            self.invalid_attempts_after_latest_action.append(
                PlannerInvalidAttempt(
                    action=step.action,
                    purpose=step.purpose,
                    policy_rule=step.policy_rule,
                    result_summary=step.result_summary,
                    evaluator_feedback=step.evaluator_result.evaluator_feedback,
                )
            )
            return

        action = PlannerActionHistory(
            action=step.action,
            purpose=step.purpose,
            policy_rule=step.policy_rule,
            result_candidate_ids=step.result_candidate_ids,
            result_groups=_result_groups_for_step(step),
            result_summary=step.result_summary,
            watched_segments=step.watched_segments,
            evaluator_feedback=step.evaluator_result.evaluator_feedback,
            irrelevant_reason=step.evaluator_result.irrelevant_reason,
            qa_result=step.qa_result,
        )
        _compact_action_results(action)
        if self.latest_valid_action is not None:
            self.used_actions.append(self.latest_valid_action)
        self.latest_valid_action = action
        self.invalid_attempts_after_latest_action = []
        self._merge_keyframe_evidence(step)
        self._merge_wrongly_submitted_candidates(step)
        self._merge_submitted_candidates(step)

    def include_seen_keyframes(
        self,
        candidate_ids: list[str],
        *,
        returned_candidate_count: int,
        new_candidate_count: int | None = None,
    ) -> None:
        """Associate cached known keyframes with the latest returning action."""

        action = self.latest_valid_action
        if action is None:
            return

        evidence_by_id = {
            evidence.keyframe_id: evidence for evidence in self.keyframe_evidence
        }
        known_irrelevant_ids = {
            candidate_id
            for history_action in self.used_actions
            for candidate_id in history_action.result_groups.irrelevant
        }
        cached_ids = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in evidence_by_id
        ]
        cached_irrelevant_ids = [
            candidate_id
            for candidate_id in candidate_ids
            if (
                candidate_id in known_irrelevant_ids
                and candidate_id not in evidence_by_id
                and candidate_id not in action.result_groups.relevant
                and candidate_id not in action.result_groups.highly_relevant
            )
        ]
        retained_ids = _unique([*(action.result_candidate_ids or []), *cached_ids])
        if retained_ids:
            action.result_candidate_ids = retained_ids

        for candidate_id in cached_ids:
            evidence = evidence_by_id[candidate_id]
            if evidence.category == "highly_relevant":
                action.result_groups.highly_relevant = _unique(
                    [*action.result_groups.highly_relevant, candidate_id]
                )
                action.result_groups.relevant = [
                    item
                    for item in action.result_groups.relevant
                    if item != candidate_id
                ]
            elif candidate_id not in action.result_groups.highly_relevant:
                action.result_groups.relevant = _unique(
                    [*action.result_groups.relevant, candidate_id]
                )

        if cached_irrelevant_ids:
            action.result_groups.irrelevant = _unique(
                [*action.result_groups.irrelevant, *cached_irrelevant_ids]
            )

        if new_candidate_count == 0:
            candidate_noun = (
                "candidate" if returned_candidate_count == 1 else "candidates"
            )
            action.result_summary = (
                f"The retrieval returned {returned_candidate_count} previously seen "
                f"{candidate_noun} and 0 new candidates."
            )
            return

        keyframe_noun = "keyframe" if returned_candidate_count == 1 else "keyframes"
        relevant_verb = "was" if len(retained_ids) == 1 else "were"
        irrelevant_count = len(action.result_groups.irrelevant)
        irrelevant_verb = "was" if irrelevant_count == 1 else "were"
        action.result_summary = (
            f"Out of {returned_candidate_count} {keyframe_noun}, "
            f"{len(retained_ids)} {relevant_verb} relevant and "
            f"{irrelevant_count} {irrelevant_verb} irrelevant."
        )

    def action_strings(self, *, include_invalid: bool = True) -> list[str]:
        actions = [action.action for action in self.used_actions]
        if self.latest_valid_action is not None:
            actions.append(self.latest_valid_action.action)
        if include_invalid:
            actions.extend(
                attempt.action for attempt in self.invalid_attempts_after_latest_action
            )
        return actions

    def __len__(self) -> int:
        return len(self._legacy_steps())

    def __iter__(self):
        return iter(self._legacy_steps())

    def __getitem__(self, index: int) -> PlannerHistoryStep:
        return self._legacy_steps()[index]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, list):
            return self._legacy_steps() == other
        return super().__eq__(other)

    def _merge_keyframe_evidence(self, step: PlannerHistoryStep) -> None:
        watched_by_id = {
            segment.candidate_id: segment for segment in step.watched_segments or []
        }
        evidence_groups: tuple[
            tuple[
                Literal["relevant", "highly_relevant"],
                list[PlannerEvaluatedImage],
            ],
            ...,
        ] = (
            ("highly_relevant", step.evaluator_result.highly_relevant_images),
            ("relevant", step.evaluator_result.relevant_images),
        )
        for category, images in evidence_groups:
            for image in images:
                segment = watched_by_id.get(image.image_id)
                existing = next(
                    (
                        evidence
                        for evidence in self.keyframe_evidence
                        if evidence.keyframe_id == image.image_id
                    ),
                    None,
                )
                if existing is None:
                    self.keyframe_evidence.append(
                        PlannerKeyframeEvidence(
                            category=category,
                            keyframe_id=image.image_id,
                            video_id=image.video_id
                            or _video_id_from_candidate_id(image.image_id),
                            timestamp_ms=image.timestamp_ms,
                            frame_description=image.description,
                            scene_description=image.scene_description,
                            reason=image.reason,
                            inspected_segment=segment,
                            visual_similarity=image.visual_similarity,
                            temporal_status=image.temporal_status,
                        )
                    )
                    continue

                if existing.category == "relevant" and category == "highly_relevant":
                    existing.category = "highly_relevant"
                    existing.reason = image.reason
                if image.description:
                    existing.frame_description = image.description
                if image.scene_description:
                    existing.scene_description = image.scene_description
                if existing.video_id is None:
                    existing.video_id = image.video_id or _video_id_from_candidate_id(
                        image.image_id
                    )
                if image.timestamp_ms is not None:
                    existing.timestamp_ms = image.timestamp_ms
                if segment is not None:
                    existing.inspected_segment = segment
                if image.visual_similarity not in {None, VisualSimilarity.UNKNOWN}:
                    existing.visual_similarity = image.visual_similarity
                if image.temporal_status not in {None, TemporalStatus.NOT_APPLICABLE}:
                    existing.temporal_status = image.temporal_status

    def _merge_wrongly_submitted_candidates(self, step: PlannerHistoryStep) -> None:
        existing = {
            (candidate.video_id, candidate.timestamp_ms): candidate
            for candidate in self.wrongly_submitted_candidates
        }
        for candidate in step.wrongly_submitted_candidates:
            key = (candidate.video_id, candidate.timestamp_ms)
            previous = existing.get(key)
            if previous is not None:
                if (
                    previous.scene_start_ms is None
                    and candidate.scene_start_ms is not None
                ):
                    previous.scene_start_ms = candidate.scene_start_ms
                    previous.scene_end_ms = candidate.scene_end_ms
                continue
            self.wrongly_submitted_candidates.append(candidate)
            existing[key] = candidate

    def _merge_submitted_candidates(self, step: PlannerHistoryStep) -> None:
        existing = {
            (candidate.video_id, candidate.timestamp_ms, candidate.status)
            for candidate in self.submitted_candidates
        }
        for candidate in step.submitted_candidates:
            key = (candidate.video_id, candidate.timestamp_ms, candidate.status)
            if key in existing:
                continue
            self.submitted_candidates.append(candidate)
            existing.add(key)

    def _legacy_steps(self) -> list[PlannerHistoryStep]:
        steps = [
            PlannerHistoryStep(
                action=action.action,
                purpose=action.purpose,
                policy_rule=action.policy_rule,
                action_valid=True,
                result_candidate_ids=action.result_candidate_ids,
                result_summary=action.result_summary,
                watched_segments=action.watched_segments,
                evaluator_result=_legacy_evaluator_result(
                    action,
                    self.keyframe_evidence,
                ),
                qa_result=action.qa_result,
            )
            for action in self.used_actions
        ]
        if self.latest_valid_action is not None:
            steps.append(
                PlannerHistoryStep(
                    action=self.latest_valid_action.action,
                    purpose=self.latest_valid_action.purpose,
                    policy_rule=self.latest_valid_action.policy_rule,
                    action_valid=True,
                    result_candidate_ids=self.latest_valid_action.result_candidate_ids,
                    result_summary=self.latest_valid_action.result_summary,
                    watched_segments=self.latest_valid_action.watched_segments,
                    evaluator_result=_legacy_evaluator_result(
                        self.latest_valid_action,
                        self.keyframe_evidence,
                    ),
                    qa_result=self.latest_valid_action.qa_result,
                )
            )
        steps.extend(
            PlannerHistoryStep(
                action=attempt.action,
                purpose=attempt.purpose,
                policy_rule=attempt.policy_rule,
                action_valid=False,
                result_summary=attempt.result_summary,
                evaluator_result=PlannerEvaluatorResult(
                    evaluator_feedback=attempt.evaluator_feedback,
                ),
            )
            for attempt in self.invalid_attempts_after_latest_action
        )
        return steps


class PlannerRequest(BaseModel):
    """Input for selecting the next retrieval action."""

    encoded_task: EncodedTask = Field(
        description="Task representation produced by the task encoder.",
    )
    history: PlannerCondensedHistory = Field(
        default_factory=PlannerCondensedHistory,
        description="History of previous actions and candidate assessments.",
    )
    history_images: list[ImageInput] = Field(
        default_factory=list,
        description=(
            "Keyframes rated relevant or highly relevant in the latest retrieval "
            "action that returned keyframes. These are attached to the planner "
            "prompt as ordered visual references."
        ),
    )
    available_actions: list[PlannerActionSpec] = Field(
        default_factory=default_available_actions,
        min_length=1,
        description="Actions the planner is allowed to emit.",
    )
    dataset: Any | None = Field(
        default=None,
        description=(
            "Dataset identifier used for parser/context plumbing. This is not "
            "included in the planner prompt when encoded_task already carries it."
        ),
    )
    task_type: Any | None = Field(
        default=None,
        description=(
            "Task type used for prompt context plumbing. This is not included "
            "in the planner prompt when encoded_task already carries it."
        ),
    )
    task_videos: list[VideoInput] = Field(
        default_factory=list,
        description="Task/reference videos allowed as planner target context.",
    )
    submission_ratings_visible: bool = Field(
        default=True,
        description=(
            "Whether AVS submission outcomes are present in planner history. False "
            "means the planner sees only which candidates were submitted."
        ),
        exclude=True,
    )
    task: TaskCreate | None = Field(
        default=None,
        description=(
            "Deprecated compatibility field for older callers. The original task "
            "payload is never included in the planner prompt."
        ),
        exclude=True,
    )

    @model_validator(mode="after")
    def _require_context_fields(self) -> "PlannerRequest":
        """Require dataset/task_type through encoded_task, explicit fields, or legacy task."""

        if (
            _first_available(self.dataset, self.task, self.encoded_task, attr="dataset")
            is None
        ):
            raise ValueError(
                "PlannerRequest requires dataset via dataset, encoded_task.dataset, or legacy task.dataset."
            )
        task_type = _first_available(
            self.task_type, self.task, self.encoded_task, attr="task_type"
        )
        if task_type is None:
            raise ValueError(
                "PlannerRequest requires task_type via task_type, encoded_task.task_type, or legacy task.task_type."
            )
        task_type_value = getattr(task_type, "value", task_type)
        if isinstance(task_type_value, str) and task_type_value.lower() == "avs":
            self.available_actions = [
                action
                for action in self.available_actions
                if action.code.lower() != "an"
            ]
            if not self.available_actions:
                raise ValueError("AVS planner requests require a non-AN action.")
        return self

    @computed_field
    @property
    def allowed_action_codes(self) -> list[str]:
        """Action tokens accepted for this planning request."""

        return [action.code.lower() for action in self.available_actions]


class PlannerDecision(BaseModel):
    """Validated planner output."""

    action: str = Field(
        min_length=1,
        description="Single retrieval action string to execute next.",
    )
    purpose: str | None = Field(
        default=None,
        min_length=1,
        description="Short explanation of what evidence the action should obtain.",
    )
    policy_rule: str | None = Field(
        default=None,
        min_length=1,
        description="Validated action-selection policy rule used for the decision.",
    )
    query: PlannerOperation = Field(
        description="Parsed retrieval query or question-answering operation.",
    )
    endpoint: str = Field(description="Data-layer endpoint for the parsed query.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-essential metadata about parsing or planning.",
    )


def _first_available(initial: Any, *objects: Any, attr: str) -> Any | None:
    if initial is not None:
        return initial

    for obj in objects:
        if obj is None:
            continue
        value = getattr(obj, attr, None)
        if value is not None:
            return value

    return None


def _result_groups_for_step(step: PlannerHistoryStep) -> PlannerResultGroups:
    result = step.evaluator_result
    return PlannerResultGroups(
        relevant=_unique(image.image_id for image in result.relevant_images),
        irrelevant=_unique(
            image_id
            for group in result.irrelevant_groups
            for image_id in group.image_ids
        ),
        highly_relevant=_unique(
            image.image_id for image in result.highly_relevant_images
        ),
    )


def _compact_action_results(action: PlannerActionHistory) -> None:
    """Keep compact action-to-keyframe ids without duplicating evidence text."""

    retained_ids = _unique(
        action.result_groups.relevant + action.result_groups.highly_relevant
    )
    original_ids = action.result_candidate_ids
    if original_ids is not None:
        original_count = len(original_ids)
        retained_set = set(retained_ids)
        action.result_candidate_ids = [
            candidate_id
            for candidate_id in original_ids
            if candidate_id in retained_set
        ]
        action.result_candidate_ids.extend(
            candidate_id
            for candidate_id in retained_ids
            if candidate_id not in action.result_candidate_ids
        )

        if action.result_summary and action.result_summary.startswith("Returned "):
            words = action.result_summary.split()
            try:
                candidate_count = int(words[1])
            except (IndexError, ValueError):
                candidate_count = original_count
            keyframe_noun = "keyframe" if candidate_count == 1 else "keyframes"
            relevant_verb = "was" if len(retained_ids) == 1 else "were"
            irrelevant_count = max(candidate_count - len(retained_ids), 0)
            irrelevant_verb = "was" if irrelevant_count == 1 else "were"
            action.result_summary = (
                f"Out of {candidate_count} {keyframe_noun}, "
                f"{len(retained_ids)} {relevant_verb} relevant and "
                f"{irrelevant_count} {irrelevant_verb} irrelevant."
            )

    action.irrelevant_reason = None
    if action.watched_segments is not None:
        retained_set = set(retained_ids)
        action.watched_segments = [
            segment
            for segment in action.watched_segments
            if segment.candidate_id in retained_set
        ]


def _scene_reference(
    keyframe_id: str,
    watched_by_id: dict[str, PlannerWatchedSegment],
) -> dict[str, Any]:
    video_id = _video_id_from_candidate_id(keyframe_id)
    segment = watched_by_id.get(keyframe_id)
    if segment is None:
        return {
            "video_id": video_id,
            "start_ms": None,
            "end_ms": None,
        }

    return {
        "video_id": video_id,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
    }


def _legacy_evaluator_result(
    action: PlannerActionHistory,
    evidence: list[PlannerKeyframeEvidence],
) -> PlannerEvaluatorResult:
    return PlannerEvaluatorResult(
        relevant_images=[
            _legacy_evaluated_image(keyframe_id, evidence)
            for keyframe_id in action.result_groups.relevant
        ],
        highly_relevant_images=[
            _legacy_evaluated_image(keyframe_id, evidence)
            for keyframe_id in action.result_groups.highly_relevant
        ],
        irrelevant_groups=(
            [
                PlannerIrrelevantImageGroup(
                    image_ids=action.result_groups.irrelevant,
                )
            ]
            if action.result_groups.irrelevant
            else []
        ),
        evaluator_feedback=action.evaluator_feedback,
        irrelevant_reason=action.irrelevant_reason,
    )


def _legacy_evaluated_image(
    keyframe_id: str,
    evidence: list[PlannerKeyframeEvidence],
) -> PlannerEvaluatedImage:
    keyframe_evidence = next(
        (item for item in evidence if keyframe_id == item.keyframe_id),
        None,
    )
    if keyframe_evidence is None:
        return PlannerEvaluatedImage(
            image_id=keyframe_id,
            description=f"Keyframe evidence for {keyframe_id}.",
            reason="This keyframe was grouped as useful evidence.",
        )
    return PlannerEvaluatedImage(
        image_id=keyframe_id,
        description=keyframe_evidence.frame_description,
        scene_description=keyframe_evidence.scene_description,
        video_id=keyframe_evidence.video_id,
        timestamp_ms=keyframe_evidence.timestamp_ms,
        reason=keyframe_evidence.reason,
        visual_similarity=keyframe_evidence.visual_similarity,
        temporal_status=keyframe_evidence.temporal_status,
    )


def _is_legacy_invalid_history_step(step: PlannerHistoryStep) -> bool:
    summary = step.result_summary or ""
    return summary.startswith(
        (
            "This action was invalid:",
            "The previous action output was invalid:",
        )
    )


def _video_id_from_candidate_id(candidate_id: str) -> str | None:
    parts = candidate_id.rsplit("_", 1)
    if len(parts) != 2 or not all(parts):
        return None
    return parts[0]


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
