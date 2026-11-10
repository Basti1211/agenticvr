"""Browser boundary and VLM-backed frame/scene evidence analysis."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from application.settings.settings import _browser_settings
from application.system.browser.prompt import (
    BROWSER_FRAME_ANALYSIS_CONTEXT_PROMPT,
    BROWSER_FRAME_ANALYSIS_SYSTEM_PROMPT,
    BROWSER_SCENE_ANALYSIS_CONTEXT_PROMPT,
    BROWSER_SCENE_ANALYSIS_SYSTEM_PROMPT,
)
from application.system.prompts.template import render_prompt
from application.system.task_input_access import remove_task_question
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.executor import run, run_parallel
from application.system.vlm.prompt_trace import prompt_trace
from application.types.browser.models import (
    BrowserFrameAnalysis,
    BrowserFrameAnalysisOutput,
    BrowserMediaCandidate,
    BrowserRequest,
    BrowserSceneAnalysis,
    BrowserSceneAnalysisOutput,
    KeyframeDecision,
    SceneVerdict,
)
from application.types.vlm.prompt import (
    ImageInput,
    LlmAgent,
    Prompt,
    PromptContext,
)


VlmRunner = Callable[[str, Prompt], Awaitable[str]]
VlmParallelRunner = Callable[[str, list[Prompt]], Awaitable[list[str]]]
BrowserCacheKey = tuple[str, str, str, str, str]
BROWSER_CACHE_MAX_ENTRIES = 10_000
logger = logging.getLogger(__name__)


class BrowserSettings:
    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        self.max_tokens = max_tokens

    @classmethod
    def load(cls) -> "BrowserSettings":
        settings = _browser_settings()
        return cls(max_tokens=int(settings["max_tokens"]))


class Browser(ABC):
    """Produces static frame evidence and temporal scene evidence."""

    @abstractmethod
    async def analyze_frames(
        self,
        request: BrowserRequest,
    ) -> BrowserFrameAnalysisOutput:
        """Describe and assess each supplied frame in one call per candidate."""

    @abstractmethod
    async def analyze_scenes(
        self,
        request: BrowserRequest,
    ) -> BrowserSceneAnalysisOutput:
        """Describe and assess each supplied video clip in one call per candidate."""


class PromptBrowser(Browser):
    """VLM-backed implementation of the two-stage browser contract."""

    def __init__(
        self,
        model: str = "qwen",
        runner: VlmRunner = run,
        parallel_runner: VlmParallelRunner | None = None,
        settings: BrowserSettings | None = None,
    ) -> None:
        self.model = model
        self.runner = runner
        if parallel_runner is not None:
            self.parallel_runner = parallel_runner
        elif runner is run:
            self.parallel_runner = run_parallel
        else:
            self.parallel_runner = None
        self.settings = settings or BrowserSettings.load()
        self._frame_cache: OrderedDict[BrowserCacheKey, BrowserFrameAnalysis] = (
            OrderedDict()
        )
        self._scene_cache: OrderedDict[BrowserCacheKey, BrowserSceneAnalysis] = (
            OrderedDict()
        )

    async def analyze_frames(
        self,
        request: BrowserRequest,
    ) -> BrowserFrameAnalysisOutput:
        if any(candidate.video is not None for candidate in request.candidates):
            raise ValueError("Frame analysis requires image evidence.")

        total_start = timer_start()
        results: list[BrowserFrameAnalysis | None] = [None] * len(request.candidates)
        pending_by_key: OrderedDict[
            BrowserCacheKey,
            list[tuple[int, BrowserMediaCandidate]],
        ] = OrderedDict()
        cache_hits: list[dict[str, str]] = []
        for index, candidate in enumerate(request.candidates):
            cache_key = self._cache_key(request, candidate)
            cached = self._cache_get(self._frame_cache, cache_key)
            if cached is None:
                pending_by_key.setdefault(cache_key, []).append((index, candidate))
                continue
            results[index] = cached.model_copy(
                update={"candidate_id": candidate.identifier}
            )
            cache_hits.append(
                {
                    "candidate_id": candidate.identifier,
                    "decision": cached.decision.value,
                }
            )

        pending = list(pending_by_key.items())
        llm_calls: list[dict[str, Any]] = []
        last_errors: dict[str, ValueError] = {}
        for attempt in (1, 2):
            if not pending:
                break
            prompts = [
                build_frame_analysis_prompt(request, members[0][1], self.settings)
                for _, members in pending
            ]
            started = timer_start()
            responses = await self._run_parallel(prompts)
            batch_duration_ms = elapsed_ms(started)
            retry: list[
                tuple[BrowserCacheKey, list[tuple[int, BrowserMediaCandidate]]]
            ] = []
            for (cache_key, members), prompt, response in zip(
                pending,
                prompts,
                responses,
                strict=True,
            ):
                candidate = members[0][1]
                call = _llm_call(
                    candidate,
                    members,
                    prompt,
                    response,
                    self.model,
                    attempt,
                    batch_duration_ms,
                    len(prompts),
                )
                try:
                    parsed = parse_frame_analysis_output(
                        response,
                        candidate.identifier,
                    )
                except ValueError as error:
                    call["error"] = str(error)
                    llm_calls.append(call)
                    last_errors[candidate.identifier] = error
                    retry.append((cache_key, members))
                    continue

                call["decision"] = parsed.decision.value
                llm_calls.append(call)
                self._cache_put(self._frame_cache, cache_key, parsed)
                for index, member in members:
                    results[index] = parsed.model_copy(
                        update={"candidate_id": member.identifier}
                    )
            pending = retry

        fallback_ids: list[str] = []
        for _, members in pending:
            for index, member in members:
                fallback_ids.append(member.identifier)
                results[index] = BrowserFrameAnalysis(
                    candidate_id=member.identifier,
                    frame_description="Frame analysis was unavailable.",
                    decision=KeyframeDecision.DISCARD,
                    possible_target_moment=None,
                    visible_support=[],
                    visible_conflicts=[],
                    reason=str(
                        last_errors.get(
                            members[0][1].identifier,
                            "The browser returned invalid structured output.",
                        )
                    ),
                )

        output = BrowserFrameAnalysisOutput(
            results=[result for result in results if result is not None],
            metadata={
                "llm_calls": llm_calls,
                "cache_hits": cache_hits,
                "deduplicated_candidate_count": sum(
                    len(members) - 1 for members in pending_by_key.values()
                ),
                "invalid_output_fallback_candidate_ids": fallback_ids,
            },
        )
        logger.info(
            "timing.browser.analyze_frames.done candidate_count=%s total_ms=%.3f",
            len(request.candidates),
            elapsed_ms(total_start),
        )
        return output

    async def analyze_scenes(
        self,
        request: BrowserRequest,
    ) -> BrowserSceneAnalysisOutput:
        if any(candidate.video is None for candidate in request.candidates):
            raise ValueError("Scene analysis requires video evidence.")

        total_start = timer_start()
        results: list[BrowserSceneAnalysis | None] = [None] * len(request.candidates)
        pending_by_key: OrderedDict[
            BrowserCacheKey,
            list[tuple[int, BrowserMediaCandidate]],
        ] = OrderedDict()
        cache_hits: list[dict[str, str]] = []
        for index, candidate in enumerate(request.candidates):
            cache_key = self._cache_key(request, candidate)
            cached = self._cache_get(self._scene_cache, cache_key)
            if cached is None:
                pending_by_key.setdefault(cache_key, []).append((index, candidate))
                continue
            results[index] = cached.model_copy(
                update={"candidate_id": candidate.identifier}
            )
            cache_hits.append(
                {
                    "candidate_id": candidate.identifier,
                    "verdict": cached.verdict.value,
                }
            )

        pending = list(pending_by_key.items())
        llm_calls: list[dict[str, Any]] = []
        last_errors: dict[str, ValueError] = {}
        for attempt in (1, 2):
            if not pending:
                break
            prompts = [
                build_scene_analysis_prompt(request, members[0][1], self.settings)
                for _, members in pending
            ]
            started = timer_start()
            responses = await self._run_parallel(prompts)
            batch_duration_ms = elapsed_ms(started)
            retry: list[
                tuple[BrowserCacheKey, list[tuple[int, BrowserMediaCandidate]]]
            ] = []
            for (cache_key, members), prompt, response in zip(
                pending,
                prompts,
                responses,
                strict=True,
            ):
                candidate = members[0][1]
                call = _llm_call(
                    candidate,
                    members,
                    prompt,
                    response,
                    self.model,
                    attempt,
                    batch_duration_ms,
                    len(prompts),
                )
                try:
                    parsed = parse_scene_analysis_output(
                        response,
                        candidate.identifier,
                    )
                except ValueError as error:
                    call["error"] = str(error)
                    llm_calls.append(call)
                    last_errors[candidate.identifier] = error
                    retry.append((cache_key, members))
                    continue

                call["verdict"] = parsed.verdict.value
                llm_calls.append(call)
                self._cache_put(self._scene_cache, cache_key, parsed)
                for index, member in members:
                    results[index] = parsed.model_copy(
                        update={"candidate_id": member.identifier}
                    )
            pending = retry

        fallback_ids: list[str] = []
        for _, members in pending:
            for index, member in members:
                fallback_ids.append(member.identifier)
                results[index] = BrowserSceneAnalysis(
                    candidate_id=member.identifier,
                    scene_description="Scene analysis was unavailable.",
                    verdict=SceneVerdict.RELEVANT,
                    matching_evidence=[],
                    missing_evidence=["Scene analysis was unavailable."],
                    contradicting_evidence=[],
                    best_match_start_seconds=None,
                    best_match_end_seconds=None,
                    reason=str(
                        last_errors.get(
                            members[0][1].identifier,
                            "The browser returned invalid structured output.",
                        )
                    ),
                )

        output = BrowserSceneAnalysisOutput(
            results=[result for result in results if result is not None],
            metadata={
                "llm_calls": llm_calls,
                "cache_hits": cache_hits,
                "deduplicated_candidate_count": sum(
                    len(members) - 1 for members in pending_by_key.values()
                ),
                "invalid_output_fallback_candidate_ids": fallback_ids,
            },
        )
        logger.info(
            "timing.browser.analyze_scenes.done candidate_count=%s total_ms=%.3f",
            len(request.candidates),
            elapsed_ms(total_start),
        )
        return output

    def _cache_key(
        self,
        request: BrowserRequest,
        candidate: BrowserMediaCandidate,
    ) -> BrowserCacheKey:
        task_type = request.task_type.value if request.task_type is not None else ""
        dataset = request.dataset.value if request.dataset is not None else ""
        return (
            self.model,
            request.task_description,
            task_type,
            dataset,
            _candidate_evidence_key(candidate, dataset),
        )

    @staticmethod
    def _cache_get(cache: OrderedDict, cache_key: BrowserCacheKey):
        value = cache.get(cache_key)
        if value is not None:
            cache.move_to_end(cache_key)
        return value

    @staticmethod
    def _cache_put(cache: OrderedDict, cache_key: BrowserCacheKey, value: Any) -> None:
        cache[cache_key] = value
        cache.move_to_end(cache_key)
        while len(cache) > BROWSER_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)

    async def _run(self, prompt: Prompt) -> str:
        return await self.runner(self.model, prompt)

    async def _run_parallel(self, prompts: list[Prompt]) -> list[str]:
        if not prompts:
            return []
        if self.parallel_runner is None:
            return await asyncio.gather(*(self._run(prompt) for prompt in prompts))
        return await self.parallel_runner(self.model, prompts)


def build_frame_analysis_prompt(
    request: BrowserRequest,
    candidate: BrowserMediaCandidate,
    settings: BrowserSettings | None = None,
) -> Prompt:
    """Build the single authoritative frame-analysis prompt."""

    if candidate.video is not None:
        raise ValueError("Frame analysis requires image evidence.")
    settings = settings or BrowserSettings.load()
    return Prompt(
        system_prompt=render_prompt(
            BROWSER_FRAME_ANALYSIS_SYSTEM_PROMPT,
            request.task_type,
            request.dataset,
        ),
        context_prompt=render_prompt(
            BROWSER_FRAME_ANALYSIS_CONTEXT_PROMPT,
            request.task_type,
            request.dataset,
        ),
        max_tokens=settings.max_tokens,
        task_text=_task_text(request),
        task_images=request.task_images,
        task_videos=request.task_videos,
        context_images=_candidate_images([candidate]),
        context_videos=[],
        context=PromptContext(
            llm_agent=LlmAgent.BROWSER_CLASSIFIER,
            dataset=request.dataset,
            task_type=request.task_type,
        ),
    )


def build_scene_analysis_prompt(
    request: BrowserRequest,
    candidate: BrowserMediaCandidate,
    settings: BrowserSettings | None = None,
) -> Prompt:
    """Build the temporal scene-analysis prompt for a visually strong candidate."""

    if candidate.video is None:
        raise ValueError("Scene analysis requires video evidence.")
    settings = settings or BrowserSettings.load()
    context_prompt = render_prompt(
        BROWSER_SCENE_ANALYSIS_CONTEXT_PROMPT,
        request.task_type,
        request.dataset,
    )
    start_ms = candidate.metadata.get("time_scene_start")
    end_ms = candidate.metadata.get("time_scene_end")
    if isinstance(start_ms, int) and isinstance(end_ms, int) and end_ms > start_ms:
        context_prompt += (
            "\nThe attached candidate clip lasts "
            f"{(end_ms - start_ms) / 1000:.3f} seconds; relative time 0 is its "
            "first frame.\n"
        )
    return Prompt(
        system_prompt=render_prompt(
            BROWSER_SCENE_ANALYSIS_SYSTEM_PROMPT,
            request.task_type,
            request.dataset,
        ),
        context_prompt=context_prompt,
        max_tokens=settings.max_tokens,
        task_text=_task_text(request),
        task_images=request.task_images,
        task_videos=request.task_videos,
        context_images=[],
        context_videos=[candidate.video],
        context=PromptContext(
            llm_agent=LlmAgent.BROWSER_CLASSIFIER,
            dataset=request.dataset,
            task_type=request.task_type,
        ),
    )


def parse_frame_analysis_output(
    response: str,
    candidate_id: str,
) -> BrowserFrameAnalysis:
    """Parse the strict frame-analysis JSON contract."""

    payload = _parse_json_object(response, "Frame analysis")
    _require_exact_fields(
        payload,
        {
            "frame_description",
            "decision",
            "possible_target_moment",
            "visible_support",
            "visible_conflicts",
            "reason",
        },
        "Frame analysis",
    )
    try:
        return BrowserFrameAnalysis(candidate_id=candidate_id, **payload)
    except (TypeError, ValidationError) as error:
        raise ValueError("Frame analysis contains invalid field values.") from error


def parse_scene_analysis_output(
    response: str,
    candidate_id: str,
) -> BrowserSceneAnalysis:
    """Parse the strict scene-analysis JSON contract."""

    payload = _parse_json_object(response, "Scene analysis")
    _require_exact_fields(
        payload,
        {
            "scene_description",
            "verdict",
            "matching_evidence",
            "missing_evidence",
            "contradicting_evidence",
            "best_match_start_seconds",
            "best_match_end_seconds",
            "reason",
        },
        "Scene analysis",
    )
    try:
        return BrowserSceneAnalysis(candidate_id=candidate_id, **payload)
    except (TypeError, ValidationError) as error:
        raise ValueError("Scene analysis contains invalid field values.") from error


def _task_text(request: BrowserRequest) -> str | None:
    return remove_task_question(request.task_description) or None


def _parse_json_object(response: str, label: str) -> dict[str, Any]:
    response_text = _unwrap_json_code_fence(response)
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must return exactly one JSON object.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must return exactly one JSON object.")
    return payload


def _unwrap_json_code_fence(response: str) -> str:
    """Accept one optional JSON code fence without accepting surrounding prose."""

    stripped = response.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if (
        len(lines) < 3
        or lines[0].strip().lower() not in {"```", "```json"}
        or lines[-1].strip() != "```"
    ):
        return stripped

    body = "\n".join(lines[1:-1]).strip()
    if "```" in body:
        return stripped
    return body


def _require_exact_fields(
    payload: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} fields do not match the contract; "
            f"missing={missing}, extra={extra}."
        )


def _candidate_images(candidates: list[BrowserMediaCandidate]) -> list[ImageInput]:
    images: list[ImageInput] = []
    for candidate in candidates:
        if candidate.images:
            images.extend(candidate.images)
        elif candidate.image is not None:
            images.append(candidate.image)
    return images


def _candidate_evidence_key(
    candidate: BrowserMediaCandidate,
    dataset: str,
) -> str:
    if candidate.video is not None:
        candidate_id = candidate.metadata.get("id")
        start_ms = candidate.metadata.get("time_scene_start")
        end_ms = candidate.metadata.get("time_scene_end")
        if (
            isinstance(candidate_id, list)
            and candidate_id
            and isinstance(candidate_id[0], str)
            and isinstance(start_ms, int)
            and isinstance(end_ms, int)
        ):
            return json.dumps(
                ["scene", dataset, candidate_id[0], start_ms, end_ms],
                ensure_ascii=True,
                separators=(",", ":"),
            )
        return f"video:{candidate.video.file.resolve()}"
    if candidate.image is not None:
        return f"image:{candidate.image.file.resolve()}"
    if candidate.images:
        return json.dumps(
            ["image_group", [str(image.file.resolve()) for image in candidate.images]],
            ensure_ascii=True,
            separators=(",", ":"),
        )
    raise ValueError("Browser candidate requires image or video evidence")


def _llm_call(
    candidate: BrowserMediaCandidate,
    members: list[tuple[int, BrowserMediaCandidate]],
    prompt: Prompt,
    response: str,
    model: str,
    attempt: int,
    batch_duration_ms: float,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.identifier,
        "candidate_ids": [member.identifier for _, member in members],
        "prompt": prompt_trace(prompt),
        "response": response,
        "model": model,
        "attempt": attempt,
        "batch_duration_ms": batch_duration_ms,
        "parallel_batch_size": batch_size,
    }
