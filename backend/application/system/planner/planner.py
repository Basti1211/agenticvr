"""Planner boundary and VLM-backed implementation."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from functools import partial
import re
from typing import Any

from application.settings.settings import _planner_settings
from application.system.planner.prompt import (
    PLANNER_CONTEXT_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from application.system.prompts.template import render_prompt
from application.system.task_context_media import (
    encoded_task_context_images,
)
from application.system.task_input_access import remove_task_question
from application.system.retrieval.endpoints import get_endpoint_for_query
from application.system.retrieval.fetch import evaluate_query
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.executor import run
from application.system.vlm.instance.openai.model import (
    ImageDetail,
    ReasoningEffort,
    run_openai,
)
from application.system.vlm.prompt_trace import prompt_trace
from application.types.planner.models import (
    ImageQAAction,
    PlannerDecision,
    PlannerRequest,
    VideoQAAction,
)
from application.types.retrieval.queries import (
    DirectAnswer,
    ImageQuery,
    IntraVideoQuery,
    LoadVideoSummary,
)
from application.types.vlm.prompt import LlmAgent, Prompt, PromptContext


VlmRunner = Callable[[str, Prompt], Awaitable[str]]
logger = logging.getLogger(__name__)


class RepeatedIntraVideoSearchError(ValueError):
    """Planner selected iv for a video that iv already searched."""


DISALLOWED_ACTIONS_AFTER = {
    "lvs": frozenset({"b", "lm"}),
    "iv": frozenset({"b", "lm"}),
}

ACTION_POLICY_RULES = {
    "nl": "initial_text_search",
    "lnl": "initial_text_search",
    "b": "candidate_refinement",
    "im": "image_similarity",
    "lvs": "video_context",
    "iv": "video_context",
    "t": "two_moment_search",
    "lm": "load_more",
    "back": "backtrack",
    "iqa": "frame_question",
    "vqa": "temporal_question",
    "an": "submit_answer",
}

CONSECUTIVE_UNSEEN_ACTIONS = frozenset(
    {"nl", "lnl", "t", "im", "b", "lvs", "iv"}
)


class PlannerSettings:
    def __init__(
        self,
        max_tokens: int,
        provider: str = "qwen",
        model: str = "qwen",
        temperature: float | None = None,
        reasoning_effort: ReasoningEffort = "low",
        api_max_output_tokens: int = 512,
        image_detail: ImageDetail = "low",
        max_attempts: int = 10,
        include_keyframe_references: bool = False,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        if provider not in {"openai", "qwen"}:
            raise ValueError("provider must be either 'openai' or 'qwen'.")
        if not model.strip():
            raise ValueError("model must not be empty.")
        if temperature is not None and temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0.")
        if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning_effort.")
        if api_max_output_tokens <= 0:
            raise ValueError("api_max_output_tokens must be greater than 0.")
        if image_detail not in {"low", "high", "auto", "original"}:
            raise ValueError("Unsupported image_detail.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        self.max_tokens = max_tokens
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.api_max_output_tokens = api_max_output_tokens
        self.image_detail = image_detail
        self.max_attempts = max_attempts
        self.include_keyframe_references = include_keyframe_references

    @classmethod
    def load(cls) -> "PlannerSettings":
        settings = _planner_settings()
        return cls(
            max_tokens=int(settings["max_tokens"]),
            provider=str(settings.get("provider", "qwen")),
            model=str(settings.get("model", "qwen")),
            temperature=(
                float(settings["temperature"])
                if settings.get("temperature") is not None
                else None
            ),
            reasoning_effort=str(settings.get("reasoning_effort", "low")),
            api_max_output_tokens=int(settings.get("api_max_output_tokens", 512)),
            image_detail=str(settings.get("image_detail", "low")),
            max_attempts=int(settings.get("max_attempts", 10)),
            include_keyframe_references=bool(
                settings.get("include_keyframe_references", False)
            ),
        )


class Planner(ABC):
    """Selects the next retrieval action from encoded task and evaluator history."""

    @abstractmethod
    async def plan(self, request: PlannerRequest) -> PlannerDecision:
        """Return one validated retrieval action."""

    @property
    def keyframe_references_enabled(self) -> bool:
        """Whether relevant retrieval keyframes should be attached to history."""

        return False


class PromptPlanner(Planner):
    """Planner implementation that asks a configured VLM for the next action."""

    def __init__(
        self,
        model: str = "qwen",
        runner: VlmRunner = run,
        max_attempts: int = 10,
        settings: PlannerSettings | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self.model = model
        self.runner = runner
        self.max_attempts = max_attempts
        self.settings = settings or PlannerSettings.load()

    @property
    def keyframe_references_enabled(self) -> bool:
        return self.settings.include_keyframe_references

    async def plan(self, request: PlannerRequest) -> PlannerDecision:
        total_start = timer_start()
        logger.info(
            "timing.planner.start model=%s history_steps=%s max_attempts=%s",
            self.model,
            len(request.history),
            self.max_attempts,
        )
        prompt, build_duration_ms = self._build_prompt(request, attempt_index=1)
        last_error: ValueError | None = None
        attempts: list[dict[str, Any]] = []
        repeated_iv_failures = 0

        for attempt_index in range(1, self.max_attempts + 1):
            response, duration_ms = await self._run_inference(
                prompt,
                attempt_index,
            )
            parse_start = timer_start()
            try:
                decision = parse_planner_decision(
                    response,
                    request,
                    require_rationale=True,
                )
                parse_duration_ms = elapsed_ms(parse_start)
                attempts.append(
                    self._attempt_metadata(
                        prompt=prompt,
                        response=response,
                        attempt_index=attempt_index,
                        duration_ms=duration_ms,
                        build_duration_ms=build_duration_ms,
                        parse_duration_ms=parse_duration_ms,
                    )
                )
                decision.metadata["attempts"] = attempts
                logger.info(
                    "timing.planner.done attempt=%s parse_ms=%.3f "
                    "total_ms=%.3f endpoint=%s action=%r",
                    attempt_index,
                    parse_duration_ms,
                    elapsed_ms(total_start),
                    decision.endpoint,
                    decision.action,
                )
                return decision
            except ValueError as error:
                parse_duration_ms = elapsed_ms(parse_start)
                last_error = error
                attempts.append(
                    self._attempt_metadata(
                        prompt=prompt,
                        response=response,
                        attempt_index=attempt_index,
                        duration_ms=duration_ms,
                        build_duration_ms=build_duration_ms,
                        parse_duration_ms=parse_duration_ms,
                        error=error,
                    )
                )
                logger.warning(
                    "timing.planner.parse.invalid attempt=%s parse_ms=%.3f error=%s",
                    attempt_index,
                    parse_duration_ms,
                    error,
                )
                if isinstance(error, RepeatedIntraVideoSearchError):
                    repeated_iv_failures += 1
                    if repeated_iv_failures >= 2:
                        repaired_decision = _repair_repeated_iv_as_global_search(
                            response,
                            request,
                        )
                        if repaired_decision is not None:
                            repaired_decision.metadata["attempts"] = attempts
                            repaired_decision.metadata["automatic_repair"] = {
                                "reason": str(error),
                                "strategy": "replace_repeated_iv_with_global_text_search",
                            }
                            logger.info(
                                "timing.planner.repaired_repeated_iv attempt=%s "
                                "total_ms=%.3f action=%r",
                                attempt_index,
                                elapsed_ms(total_start),
                                repaired_decision.action,
                            )
                            return repaired_decision
                else:
                    repeated_iv_failures = 0
                prompt, build_duration_ms = self._build_prompt(
                    request,
                    attempt_index + 1,
                    retry_feedback=self._retry_feedback(error),
                )

        logger.error(
            "timing.planner.failed attempts=%s total_ms=%.3f error=%s",
            self.max_attempts,
            elapsed_ms(total_start),
            last_error,
        )
        raise ValueError(f"Planner failed to produce a valid action: {last_error}")

    def _build_prompt(
        self,
        request: PlannerRequest,
        attempt_index: int,
        retry_feedback: str | None = None,
    ) -> tuple[Prompt, float]:
        build_start = timer_start()
        prompt = build_planner_prompt(
            request,
            retry_feedback=retry_feedback,
            settings=self.settings,
        )
        build_duration_ms = elapsed_ms(build_start)
        logger.info(
            "timing.planner.prompt_built attempt=%s build_ms=%.3f "
            "system_chars=%s context_chars=%s max_tokens=%s",
            attempt_index,
            build_duration_ms,
            len(prompt.system_prompt),
            len(prompt.context_prompt),
            prompt.max_tokens,
        )
        return prompt, build_duration_ms

    async def _run_inference(
        self,
        prompt: Prompt,
        attempt_index: int,
    ) -> tuple[str, float]:
        llm_start = timer_start()
        logger.info(
            "timing.planner.inference.start attempt=%s model=%s",
            attempt_index,
            self.model,
        )
        try:
            response = await self._run(prompt)
        except Exception:
            logger.exception(
                "timing.planner.inference.failed attempt=%s inference_ms=%.3f model=%s",
                attempt_index,
                elapsed_ms(llm_start),
                self.model,
            )
            raise
        duration_ms = elapsed_ms(llm_start)
        logger.info(
            "timing.planner.inference.done attempt=%s inference_ms=%.3f "
            "response_chars=%s",
            attempt_index,
            duration_ms,
            len(response),
        )
        return response, duration_ms

    def _attempt_metadata(
        self,
        *,
        prompt: Prompt,
        response: str,
        attempt_index: int,
        duration_ms: float,
        build_duration_ms: float,
        parse_duration_ms: float,
        error: ValueError | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "prompt": prompt_trace(prompt),
            "response": response,
            "model": self.model,
            "attempt": attempt_index,
            "duration_ms": duration_ms,
            "prompt_build_ms": build_duration_ms,
            "parse_ms": parse_duration_ms,
        }
        if error is not None:
            metadata["error"] = str(error)
        return metadata

    @staticmethod
    def _retry_feedback(error: ValueError) -> str:
        return (
            f"Previous output was invalid: {error}. "
            "Return one valid structured decision."
        )

    async def _run(self, prompt: Prompt) -> str:
        return await self.runner(self.model, prompt)


def configured_prompt_planner() -> PromptPlanner:
    """Build the planner selected in planner.json without changing other agents."""

    settings = PlannerSettings.load()
    if settings.provider == "openai":
        runner: VlmRunner = partial(
            run_openai,
            reasoning_effort=settings.reasoning_effort,
            max_output_tokens=settings.api_max_output_tokens,
            image_detail=settings.image_detail,
        )
    else:
        runner = run
    return PromptPlanner(
        model=settings.model,
        runner=runner,
        max_attempts=settings.max_attempts,
        settings=settings,
    )


def build_planner_prompt(
    request: PlannerRequest,
    retry_feedback: str | None = None,
    settings: PlannerSettings | None = None,
) -> Prompt:
    """Build a compact planner prompt that returns one retrieval action."""

    settings = settings or PlannerSettings.load()
    task_text = _encoded_task_text(request)
    task_images = encoded_task_context_images(request.encoded_task)
    task_type = _context_value(request, "task_type")
    dataset = _context_value(request, "dataset")
    history_images = (
        request.history_images if settings.include_keyframe_references else []
    )
    history = _compact_condensed_history(request.history)
    iv_searched_video_ids = sorted(_intra_video_search_video_ids(request))
    if iv_searched_video_ids:
        history["iv_searched_video_ids"] = iv_searched_video_ids
    if not _is_avs_task_type(task_type):
        history.pop("submitted_candidates", None)
    if history_images:
        history["keyframe_references"] = [
            {
                "context_image_index": index,
                "keyframe_id": image.identifier,
            }
            for index, image in enumerate(history_images, start=1)
        ]
    action_lines = [
        _format_available_action(action) for action in request.available_actions
    ]

    prior_search_evidence = (
        json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        if history or history_images
        else "none"
    )
    output_correction = (
        "\n\nOUTPUT CORRECTION:\n" + retry_feedback if retry_feedback else ""
    )

    prompt = Prompt(
        system_prompt=_with_tkis_intra_video_guidance(
            _with_keyframe_reference_guidance(
                _with_wrong_submission_guidance(
                    _with_spatial_query_guidance(
                        render_prompt(
                            PLANNER_SYSTEM_PROMPT,
                            task_type,
                            dataset,
                        )
                    ),
                    task_type=task_type,
                    submission_ratings_visible=request.submission_ratings_visible,
                ),
            ),
            task_type=task_type,
        ),
        context_prompt=_without_legacy_target_section(
            render_prompt(
                PLANNER_CONTEXT_PROMPT,
                task_type,
                dataset,
                TARGET="",
                PRIOR_SEARCH_EVIDENCE=prior_search_evidence,
                ALLOWED_ACTIONS="\n".join(action_lines),
                OUTPUT_CORRECTION=output_correction,
            )
        ),
        temperature=settings.temperature,
        max_tokens=min(settings.max_tokens, 128),
        task_text=task_text,
        task_images=task_images,
        task_videos=request.task_videos,
        context_images=history_images,
        context_videos=[],
        context=PromptContext(
            llm_agent=LlmAgent.PLANNER,
            dataset=dataset,
            task_type=task_type,
        ),
    )
    return prompt


def _encoded_task_text(request: PlannerRequest) -> str | None:
    text = getattr(request.encoded_task, "text", None)
    if isinstance(text, str) and text.strip():
        return remove_task_question(text) or None
    return None


def _without_legacy_target_section(value: str) -> str:
    """Remove the per-agent TARGET block now rendered by the shared VLM adapter."""

    return re.sub(
        r"\nTARGET:\s*\n(?=PRIOR SEARCH EVIDENCE:)",
        "\n",
        value,
        count=1,
    ).strip()


def _with_spatial_query_guidance(value: str) -> str:
    """Keep distinctive spatial constraints intact across query reformulations."""

    guidance = (
        "- Preserve every explicit spatial detail in every text-query "
        "reformulation, including left/right, foreground/background, relative "
        "positions, object counts, and layout. Do not drop these details when "
        "shortening or varying a query."
    )
    return re.sub(
        r"\n+(Return exactly one JSON object)",
        f"\n{guidance}\n\n\\1",
        value,
        count=1,
    )


def _with_wrong_submission_guidance(
    value: str,
    *,
    task_type: Any | None = None,
    submission_ratings_visible: bool = True,
) -> str:
    """Teach every planner prompt variant how to use external rejection history."""

    guidance_lines = []
    if _is_avs_task_type(task_type):
        if submission_ratings_visible:
            guidance_lines.append(
                "- Treat `submitted_candidates` as frames or scenes already submitted. "
                "Do not submit the same video/timestamp pair again, and avoid submitting "
                "another keyframe whose scene interval overlaps an already submitted "
                "scene. Accepted submissions are already correct; unavailable submissions "
                "were not classifiable from competition judgements, so search for distinct "
                "new matches."
            )
        else:
            guidance_lines.append(
                "- Treat `submitted_candidates` only as frames or scenes already "
                "submitted. Their external ratings are deliberately withheld. Do not "
                "infer an outcome, resubmit the same video/timestamp pair, or submit an "
                "overlapping scene; search for distinct new matches."
            )
    if submission_ratings_visible or not _is_avs_task_type(task_type):
        guidance_lines.append(
            "- Treat `wrongly_submitted_candidates` as definitive negative feedback "
            "from the external submission checker. Never submit the same video and "
            "timestamp pair again; exact rejected pairs are parser-invalid. When "
            "`scene_start_ms` and "
            "`scene_end_ms` are present, do not walk through adjacent timestamps in that "
            "rejected scene. A rejection does not imply that the target occurs elsewhere "
            "in the same video; return to global retrieval unless independent new evidence "
            "identifies a distinct scene."
        )
    guidance = "\n".join(guidance_lines)
    return re.sub(
        r"\n+(Return exactly one JSON object)",
        f"\n{guidance}\n\n\\1",
        value,
        count=1,
    )


def _is_avs_task_type(task_type: Any | None) -> bool:
    value = getattr(task_type, "value", task_type)
    return isinstance(value, str) and value.lower() == "avs"


def _is_tkis_task_type(task_type: Any | None) -> bool:
    value = getattr(task_type, "value", task_type)
    return isinstance(value, str) and value.lower() == "t-kis"


def _with_tkis_intra_video_guidance(
    value: str,
    *,
    task_type: Any | None,
) -> str:
    """Keep T-KIS intra-video searches focused instead of exploratory."""

    if not _is_tkis_task_type(task_type):
        return value

    guidance = (
        "LOCAL VIDEO INSPECTION POLICY:\n"
        "- `keyframe_evidence` is cumulative. An old relevant keyframe is not "
        "automatically a reason to search its video.\n"
        "- `iv` is a focused follow-up action, not a method for cross-video "
        "exploration.\n"
        "- Use `iv` only when `latest_valid_action.relevant_keyframe_ids` contains "
        "a frame with a distinctive same-video fingerprint and a specific missing "
        "target detail is likely to occur nearby. A same-video fingerprint requires "
        "identity evidence such as a distinctive landmark or layout, the same people "
        "or unusual objects, recurring text or graphics, or visible event continuity.\n"
        "- Generic overlap in subject, setting, mood, lighting, color, or common "
        "objects is not a same-video fingerprint and does not justify `iv`, even when "
        "the frame resembles one instant from the target description. Continue with "
        "global retrieval instead.\n"
        "- When comparing possible `iv` videos, prioritize unique identity linkage "
        "to the source—such as the same unusual object, people, or distinctive "
        "location—even if that keyframe shows a different phase of the event. Do not "
        "prefer a generic depiction of the target action over stronger identity "
        "evidence.\n"
        "- `temporal_status: not_verified` alone is not sufficient reason to use "
        "`iv`. Do not use `iv` merely because a video is relevant, unsearched, or "
        "present in cumulative evidence.\n"
        "- Cross-video exploration means using global retrieval such as `nl`, `t`, "
        "`im`, `b`, or `lm`, preferably with `--one-result-per-video`; do not "
        "execute `iv` separately for many videos.\n"
        "- After `iv` fails to produce substantially stronger evidence, return to "
        "global retrieval or `back` instead of immediately choosing another "
        "historical video for `iv`.\n"
        "- If one candidate satisfies all distinctive target requirements and has "
        "a known timestamp, prefer `AN` over further exploration. Do not delay "
        "submission merely to inspect additional unsearched videos."
    )
    return re.sub(
        r"\n+(Return exactly one JSON object)",
        f"\n{guidance}\n\n\\1",
        value,
        count=1,
    )


def _with_keyframe_reference_guidance(value: str) -> str:
    """Explain the ordered visual references embedded in planner history."""

    guidance = (
        "- When `keyframe_references` is present in prior search evidence, the "
        "attached context images correspond to its `context_image_index` values. "
        "Use those images as visual evidence for the associated keyframe ids.\n"
        "- Treat `visual_similarity` and `temporal_status` independently. For `b`, "
        "prefer eligible keyframes with exact or very_high visual similarity even "
        "when temporal_status is not_verified. For `iv`, not_verified alone is not "
        "a reason to inspect a video; use it only for a promising video when a "
        "focused missing detail requires temporal inspection. Treat contradicted "
        "as negative evidence, but never treat not_verified as a contradiction."
    )
    return re.sub(
        r"\n+(Return exactly one JSON object)",
        f"\n{guidance}\n\n\\1",
        value,
        count=1,
    )


def _compact_condensed_history(history: Any) -> dict[str, Any]:
    payload = history.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"metadata"},
    )
    _drop_duplicated_action_results(payload)
    compact = _remove_prompt_noise(payload)
    return _add_video_ids_to_history(compact)


def _drop_duplicated_action_results(payload: dict[str, Any]) -> None:
    """Keep complete action-to-keyframe links without duplicating evidence text."""

    actions = list(payload.get("used_actions", []))
    latest_action = payload.get("latest_valid_action")
    if isinstance(latest_action, dict):
        actions.append(latest_action)

    for action in actions:
        if not isinstance(action, dict):
            continue
        result_groups = action.pop("result_groups", None)
        if isinstance(result_groups, dict):
            relevant_ids = [
                candidate_id
                for group_name in ("relevant", "highly_relevant")
                for candidate_id in result_groups.get(group_name, [])
                if isinstance(candidate_id, str)
            ]
            action["relevant_keyframe_ids"] = list(dict.fromkeys(relevant_ids))
            irrelevant_ids = result_groups.get("irrelevant", [])
            action["irrelevant_keyframe_ids"] = (
                irrelevant_ids if isinstance(irrelevant_ids, list) else []
            )
        action.pop("watched_segments", None)
        action.pop("irrelevant_reason", None)


def _remove_prompt_noise(value: Any) -> Any:
    """Drop execution details that do not help choose the next action."""

    noisy_keys = {
        "attempt",
        "attempts",
        "duration_ms",
        "endpoint",
        "latency_ms",
        "model",
        "raw_response",
        "timing",
    }
    if isinstance(value, dict):
        history_key_aliases = {
            "evaluator_result": "feedback",
            "evaluator_feedback": "guidance",
            "highly_relevant_images": "rejected_near_matches",
            "relevant_images": "useful_near_matches",
            "irrelevant_groups": "irrelevant_patterns",
            "result_summary": "summary",
        }
        return {
            history_key_aliases.get(key, key): _remove_prompt_noise(item)
            for key, item in value.items()
            if key not in noisy_keys
        }
    if isinstance(value, list):
        return [_remove_prompt_noise(item) for item in value]
    return value


def _add_video_ids_to_history(value: Any) -> Any:
    if isinstance(value, dict):
        enriched: dict[str, Any] = {}
        for key, item in value.items():
            if key == "image_id" and isinstance(item, str):
                enriched[key] = item
                video_id = _video_id_from_candidate_id(item)
                if video_id is not None:
                    enriched["video_id"] = video_id
            elif key in {"image_ids", "candidate_ids"} and isinstance(item, list):
                enriched[key] = item
                video_ids = _video_ids_for_candidate_ids(item)
                if video_ids:
                    enriched["video_ids"] = video_ids
            else:
                enriched[key] = _add_video_ids_to_history(item)
        return enriched

    if isinstance(value, list):
        return [_add_video_ids_to_history(item) for item in value]

    return value


def _video_ids_for_candidate_ids(candidate_ids: list[Any]) -> list[str]:
    video_ids: list[str] = []
    seen: set[str] = set()
    for candidate_id in candidate_ids:
        if not isinstance(candidate_id, str):
            continue
        video_id = _video_id_from_candidate_id(candidate_id)
        if video_id is None:
            continue
        if video_id not in seen:
            video_ids.append(video_id)
            seen.add(video_id)
    return video_ids


def _video_id_from_candidate_id(candidate_id: str) -> str | None:
    parts = candidate_id.rsplit("_", 1)
    if len(parts) != 2 or not all(parts):
        return None
    return parts[0]


def _format_available_action(action: Any) -> str:
    code = getattr(action, "code", "")
    syntax = getattr(action, "syntax", "")
    description = getattr(action, "description", "")
    return f"{code}: {syntax} - {description}".strip()


def parse_planner_decision(
    response: str,
    request: PlannerRequest,
    *,
    require_rationale: bool = False,
) -> PlannerDecision:
    """Parse and validate a VLM planner response against available actions."""

    action, purpose, policy_rule = _extract_decision_fields(
        response,
        require_rationale=require_rationale,
    )
    action = _with_unseen_for_consecutive_repeat(action, request)
    action_code = _action_code(action)
    allowed_codes = set(request.allowed_action_codes)
    if action_code not in allowed_codes:
        raise ValueError(
            "Planner action must start with one of "
            f"{sorted(allowed_codes)}. Got: {action!r}"
        )
    if action_code != "lm" and _is_bare_action_code(action, action_code):
        raise ValueError(
            "Planner JSON action_call must contain the complete executable action, "
            f"including all required arguments. Bare action code: {action!r}. For "
            "example, use 'nl woman and child in a forest clearing', not 'nl'."
        )

    if action_code == "iqa":
        query = _parse_image_qa_action(action)
    elif action_code == "vqa":
        query = _parse_video_qa_action(action)
    else:
        try:
            query = evaluate_query(action, _dataset_value(request))
        except ValueError as error:
            raise ValueError(
                f"Planner action is not a valid retrieval query: {action!r}"
            ) from error

    if isinstance(query, DirectAnswer):
        known_video_ids = _known_video_ids(request)
        if query.videoId not in known_video_ids:
            raise ValueError(
                "Direct Answer video id must appear in prior planner evidence. "
                f"Unknown video id: {query.videoId!r}."
            )
    elif isinstance(query, LoadVideoSummary):
        if query.itemId not in _known_keyframe_ids(request):
            raise ValueError(
                "Load Video Summary image id must appear in prior planner evidence. "
                f"Unknown image id: {query.itemId!r}."
            )
    elif isinstance(query, ImageQAAction):
        if query.keyframe_id not in _known_keyframe_ids(request):
            raise ValueError(
                "IQA keyframe id must appear in prior planner evidence. "
                f"Unknown keyframe id: {query.keyframe_id!r}."
            )
    elif isinstance(query, VideoQAAction):
        if query.video_name not in _known_video_ids(request):
            raise ValueError(
                "VQA video name must appear in prior planner evidence. "
                f"Unknown video name: {query.video_name!r}."
            )

    _validate_action_transition(action_code, request)
    _validate_action_selection_policy(
        action=action,
        action_code=action_code,
        query=query,
        purpose=purpose,
        policy_rule=policy_rule,
        request=request,
        require_rationale=require_rationale,
    )

    metadata = {"raw_response": response}

    return PlannerDecision(
        action=action,
        purpose=purpose,
        policy_rule=policy_rule,
        query=query,
        endpoint=(
            "qa"
            if isinstance(query, (ImageQAAction, VideoQAAction))
            else get_endpoint_for_query(query)
        ),
        metadata=metadata,
    )


def _parse_image_qa_action(action: str) -> ImageQAAction:
    head, separator, question = action.partition("||")
    parts = head.split()
    if separator != "||" or len(parts) != 2 or not question.strip():
        raise ValueError("IQA action must use: IQA <keyframe_id> || <question>.")
    return ImageQAAction(
        keyframe_id=parts[1],
        question=question.strip(),
    )


def _parse_video_qa_action(action: str) -> VideoQAAction:
    head, separator, question = action.partition("||")
    parts = head.split()
    if separator != "||" or len(parts) != 4 or not question.strip():
        raise ValueError(
            "VQA action must use: VQA <video_name> <start_ms> <end_ms> || <question>."
        )
    try:
        start_ms = int(parts[2])
        end_ms = int(parts[3])
    except ValueError as error:
        raise ValueError("VQA start_ms and end_ms must be integers.") from error
    return VideoQAAction(
        video_name=parts[1],
        start_ms=start_ms,
        end_ms=end_ms,
        question=question.strip(),
    )


def _context_value(request: PlannerRequest, name: str) -> Any:
    explicit_value = getattr(request, name, None)
    if explicit_value is not None:
        return explicit_value

    encoded_value = getattr(request.encoded_task, name, None)
    if encoded_value is not None:
        return encoded_value

    legacy_task = getattr(request, "task", None)
    if legacy_task is not None:
        legacy_value = getattr(legacy_task, name, None)
        if legacy_value is not None:
            return legacy_value

    raise ValueError(f"PlannerRequest is missing required context field: {name}")


def _dataset_value(request: PlannerRequest) -> str:
    dataset = _context_value(request, "dataset")
    return str(getattr(dataset, "value", dataset))


def _extract_decision_fields(
    response: str,
    *,
    require_rationale: bool,
) -> tuple[str, str | None, str | None]:
    stripped = _strip_code_fence(response.strip())
    if not stripped:
        raise ValueError("Planner response is empty.")

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        json_action = _json_action(parsed)
        if json_action is not None:
            purpose = _normalized_optional_text(parsed.get("purpose"))
            policy_rule = _normalized_optional_text(parsed.get("policy_rule"))
            if require_rationale and (purpose is None or policy_rule is None):
                raise ValueError(
                    "Planner JSON must contain non-empty action, purpose, and "
                    "policy_rule fields."
                )
            return _normalize_action_text(json_action), purpose, policy_rule

    if isinstance(parsed, str) and not require_rationale:
        return _normalize_action_text(parsed), None, None

    if require_rationale:
        raise ValueError(
            "Planner must return exactly one JSON object with action, purpose, and "
            "policy_rule fields."
        )

    for line in stripped.splitlines():
        normalized = _normalize_action_text(line)
        if normalized:
            return normalized, None, None

    raise ValueError("Planner response did not contain an action.")


def _json_action(parsed: dict[str, Any]) -> str | None:
    action_call = parsed.get("action_call")
    if isinstance(action_call, str):
        return action_call

    for key in ("action", "retrieval_action", "next_action"):
        value = parsed.get(key)
        if isinstance(value, str):
            arguments = parsed.get("arguments")
            if not isinstance(arguments, str):
                arguments = parsed.get("query")
            if _looks_like_bare_action_code(value) and isinstance(arguments, str):
                return f"{value.strip()} {arguments.strip()}"
            return value

    query = parsed.get("query")
    if isinstance(query, str):
        return query

    return None


def _normalized_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


def _normalize_action_text(value: str) -> str:
    normalized = value.strip().strip("`'\" ")
    for prefix in ("action:", "query:", "retrieval_action:", "next_action:"):
        if normalized.lower().startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    parts = normalized.split()
    optional_flags = {
        "[--one-result-per-video]": "--one-result-per-video",
        "[--unseen]": "--unseen",
    }
    return " ".join(optional_flags.get(part.casefold(), part) for part in parts)


def _action_code(action: str) -> str:
    parts = action.split(maxsplit=1)
    if not parts:
        raise ValueError("Planner action is empty.")
    code = parts[0].lower()
    if code.endswith(":"):
        code = code[:-1]
    if code.startswith("back("):
        return "back"
    return code


def _looks_like_bare_action_code(value: str) -> bool:
    normalized = value.strip().lower().rstrip(":")
    return normalized in ACTION_POLICY_RULES


def _is_bare_action_code(action: str, action_code: str) -> bool:
    return action.strip().lower().rstrip(":") == action_code


def _with_unseen_for_consecutive_repeat(
    action: str,
    request: PlannerRequest,
) -> str:
    """Make a repeated retrieval action advance past task-scoped seen results."""

    if _action_code(action) not in CONSECUTIVE_UNSEEN_ACTIONS:
        return action
    if _has_unseen_option(action):
        return action

    latest_action = request.history.latest_valid_action
    if latest_action is None:
        return action
    if _without_unseen_option(action) != _without_unseen_option(
        latest_action.action
    ):
        return action
    return f"{action} --unseen"


def _has_unseen_option(action: str) -> bool:
    return any(part.lower() == "--unseen" for part in action.split())


def _without_unseen_option(action: str) -> str:
    return " ".join(
        part.lower() for part in action.split() if part.lower() != "--unseen"
    )


def _repair_repeated_iv_as_global_search(
    response: str,
    request: PlannerRequest,
) -> PlannerDecision | None:
    """Preserve a repeated iv query while moving it back to global retrieval."""

    if "nl" not in set(request.allowed_action_codes):
        return None
    try:
        action, _, _ = _extract_decision_fields(response, require_rationale=True)
    except ValueError:
        return None
    parts = action.split(maxsplit=2)
    if len(parts) != 3 or _action_code(action) != "iv":
        return None

    query_parts = [
        part
        for part in parts[2].split()
        if part.lower() not in {"--unseen", "--one-result-per-video"}
    ]
    if not query_parts:
        return None
    query_text = " ".join(query_parts)
    repaired_response = json.dumps(
        {
            "action_call": f"nl {query_text} --one-result-per-video --unseen",
            "purpose": (
                "Continue the same visual search globally because intra-video search "
                "has already been used for the selected video."
            ),
            "policy_rule": "initial_text_search",
        }
    )
    return parse_planner_decision(
        repaired_response,
        request,
        require_rationale=True,
    )


def _validate_action_selection_policy(
    *,
    action: str,
    action_code: str,
    query: Any,
    purpose: str | None,
    policy_rule: str | None,
    request: PlannerRequest,
    require_rationale: bool,
) -> None:
    if isinstance(query, DirectAnswer):
        normalized_video_id = _normalize_video_id_for_policy(query.videoId)
        for rejected in request.history.wrongly_submitted_candidates:
            if (
                _normalize_video_id_for_policy(rejected.video_id)
                == normalized_video_id
                and rejected.timestamp_ms == query.timestampMs
            ):
                raise ValueError(
                    "Direct Answer (`AN`) must not repeat a video/timestamp pair "
                    "rejected by the external submission checker. "
                    f"Rejected answer: {query.videoId!r} at {query.timestampMs} ms. "
                    "Choose global retrieval or a distinct evidence-backed scene."
                )

    if isinstance(query, IntraVideoQuery):
        searched_video_ids = _intra_video_search_video_ids(request)
        if _normalize_video_id_for_policy(query.videoId) in searched_video_ids:
            alternatives = (
                "global retrieval, `back`, `VQA`, or `iv` on a different video"
                if _is_avs_task_type(_context_value(request, "task_type"))
                else "global retrieval, `back`, `VQA`, `AN`, or `iv` on a different video"
            )
            raise RepeatedIntraVideoSearchError(
                "Intra-video search (`iv`) may be used at most once per video per "
                f"task. Video {query.videoId!r} has already been searched with `iv`. "
                f"Choose {alternatives}."
            )

    if isinstance(query, ImageQuery):
        eligible_ids = _image_refinement_candidate_ids(request)
        if query.candidateId not in eligible_ids:
            raise ValueError(
                "Image similarity (`im`) requires a candidate listed in prior "
                "keyframe_evidence with category relevant or highly_relevant. "
                f"Ineligible candidate id: {query.candidateId!r}."
            )

    if policy_rule is None:
        if require_rationale:
            raise ValueError("Planner decision requires a policy_rule.")
        return
    if purpose is None:
        raise ValueError("Planner decision with a policy_rule also requires a purpose.")

    expected_rule = ACTION_POLICY_RULES[action_code]
    if policy_rule != expected_rule:
        raise ValueError(
            f"Planner policy_rule for action {action_code!r} must be "
            f"{expected_rule!r}, not {policy_rule!r}."
        )


def _image_refinement_candidate_ids(request: PlannerRequest) -> set[str]:
    candidate_ids = {
        candidate_id
        for action in _valid_history_actions(request)
        for candidate_id in (
            action.result_groups.relevant
            + action.result_groups.highly_relevant
        )
    }
    candidate_ids.update(
        evidence.keyframe_id
        for evidence in request.history.keyframe_evidence
        if evidence.category in {"relevant", "highly_relevant"}
    )
    return candidate_ids


def _intra_video_search_video_ids(request: PlannerRequest) -> set[str]:
    video_ids: set[str] = set()
    for history_action in _valid_history_actions(request):
        action_parts = history_action.action.split()
        if len(action_parts) < 2:
            continue
        try:
            action_code = _action_code(history_action.action)
        except ValueError:
            continue
        if action_code == "iv":
            video_ids.add(_normalize_video_id_for_policy(action_parts[1]))
    return video_ids


def _normalize_video_id_for_policy(video_id: str) -> str:
    normalized = video_id.strip().casefold()
    if normalized.endswith(".mp4"):
        return normalized[:-4]
    return normalized


def _validate_action_transition(
    action_code: str,
    request: PlannerRequest,
) -> None:
    previous_action_code = _last_valid_action_code(request)
    if previous_action_code is None:
        return

    disallowed_actions = DISALLOWED_ACTIONS_AFTER.get(previous_action_code, ())
    if action_code in disallowed_actions:
        raise ValueError(
            f"Planner action {action_code!r} is not allowed when the most recent "
            f"valid action is {previous_action_code!r}. Choose a different action."
        )


def _last_valid_action_code(request: PlannerRequest) -> str | None:
    allowed_codes = set(request.allowed_action_codes)
    latest_action = request.history.latest_valid_action
    if latest_action is None:
        return None
    try:
        action_code = _action_code(latest_action.action)
    except ValueError:
        return None

    if action_code in allowed_codes:
        return action_code

    return None


def _known_video_ids(request: PlannerRequest) -> set[str]:
    video_ids: set[str] = set()
    for action in _valid_history_actions(request):
        for candidate_id in action.result_candidate_ids or []:
            video_id = _video_id_from_candidate_id(candidate_id)
            if video_id is not None:
                video_ids.add(video_id)

        for candidate_id in (
            action.result_groups.relevant
            + action.result_groups.irrelevant
            + action.result_groups.highly_relevant
        ):
            video_id = _video_id_from_candidate_id(candidate_id)
            if video_id is not None:
                video_ids.add(video_id)

        action_parts = action.action.split()
        if action_parts and _action_code(action.action) == "lvs":
            if len(action_parts) >= 2 and action_parts[1]:
                video_id = _video_id_from_candidate_id(action_parts[1])
                if video_id is not None:
                    video_ids.add(video_id)
        elif action_parts and _action_code(action.action) in {"iv", "an", "vqa"}:
            if len(action_parts) >= 2 and action_parts[1]:
                video_ids.add(action_parts[1])
        if action.qa_result is not None and action.qa_result.start_ms is not None:
            video_ids.add(action.qa_result.media_id)

    for evidence in request.history.keyframe_evidence:
        if evidence.video_id:
            video_ids.add(evidence.video_id)
        video_id = _video_id_from_candidate_id(evidence.keyframe_id)
        if video_id is not None:
            video_ids.add(video_id)

    return video_ids


def _known_keyframe_ids(request: PlannerRequest) -> set[str]:
    keyframe_ids: set[str] = set()
    for action in _valid_history_actions(request):
        keyframe_ids.update(action.result_candidate_ids or [])
        keyframe_ids.update(
            action.result_groups.relevant
            + action.result_groups.irrelevant
            + action.result_groups.highly_relevant
        )
        action_parts = action.action.split()
        if action_parts and _action_code(action.action) == "lvs":
            if len(action_parts) >= 2 and action_parts[1]:
                keyframe_ids.add(action_parts[1])
        if action.qa_result is not None and action.qa_result.start_ms is None:
            keyframe_ids.add(action.qa_result.media_id)
    for evidence in request.history.keyframe_evidence:
        keyframe_ids.add(evidence.keyframe_id)
    return keyframe_ids


def _valid_history_actions(request: PlannerRequest) -> list[Any]:
    actions = list(request.history.used_actions)
    if request.history.latest_valid_action is not None:
        actions.append(request.history.latest_valid_action)
    return actions


def _has_condensed_history(history: Any) -> bool:
    return bool(
        history.latest_valid_action
        or history.used_actions
        or history.invalid_attempts_after_latest_action
        or history.keyframe_evidence
        or history.vqa_searched_video_ids
        or history.rejected_vqa_answers
        or history.wrongly_submitted_candidates
        or history.submitted_candidates
    )


def _strip_code_fence(response: str) -> str:
    lines = response.strip().splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()
