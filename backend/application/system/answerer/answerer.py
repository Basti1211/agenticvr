"""Independent answerer boundary and VLM-backed implementation."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from application.settings.settings import _answerer_settings
from application.system.answerer.prompt import (
    ANSWERER_CONTEXT_PROMPT,
    ANSWERER_SYSTEM_PROMPT,
)
from application.system.prompts.template import render_prompt
from application.system.task_context_media import encoded_task_context_images
from application.system.task_input_access import remove_task_question
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.executor import run
from application.system.vlm.prompt_trace import prompt_trace
from application.types.answerer.models import AnswererOutput, AnswererRequest
from application.types.vlm.prompt import LlmAgent, Prompt, PromptContext


VlmRunner = Callable[[str, Prompt], Awaitable[str]]
logger = logging.getLogger(__name__)
_ANSWER_PATTERN = re.compile(r"^([^,\s]+)\s*,\s*(\d+)$")


class AnswererSettings:
    """Generation settings for the per-iteration answerer."""

    def __init__(self, max_tokens: int) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        self.max_tokens = max_tokens

    @classmethod
    def load(cls) -> "AnswererSettings":
        settings = _answerer_settings()
        return cls(max_tokens=int(settings["max_tokens"]))


class Answerer(ABC):
    """Produces one best-guess answer without controlling the retrieval loop."""

    @abstractmethod
    async def answer(self, request: AnswererRequest) -> AnswererOutput:
        """Return one video id and timestamp guess."""


class PromptAnswerer(Answerer):
    """Answerer implementation backed by the configured VLM."""

    def __init__(
        self,
        model: str = "qwen",
        runner: VlmRunner = run,
        max_attempts: int = 3,
        settings: AnswererSettings | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self.model = model
        self.runner = runner
        self.max_attempts = max_attempts
        self.settings = settings or AnswererSettings.load()

    async def answer(self, request: AnswererRequest) -> AnswererOutput:
        prompt = build_answerer_prompt(request, settings=self.settings)
        attempts: list[dict[str, Any]] = []
        last_error: ValueError | None = None

        for attempt_index in range(1, self.max_attempts + 1):
            inference_start = timer_start()
            response = await self.runner(self.model, prompt)
            duration_ms = elapsed_ms(inference_start)
            try:
                output = parse_answerer_output(response)
            except ValueError as error:
                last_error = error
                attempts.append(
                    {
                        "prompt": prompt_trace(prompt),
                        "response": response,
                        "error": str(error),
                        "model": self.model,
                        "attempt": attempt_index,
                        "duration_ms": duration_ms,
                    }
                )
                prompt = build_answerer_prompt(
                    request,
                    retry_feedback=str(error),
                    settings=self.settings,
                )
                continue

            attempts.append(
                {
                    "prompt": prompt_trace(prompt),
                    "response": response,
                    "model": self.model,
                    "attempt": attempt_index,
                    "duration_ms": duration_ms,
                }
            )
            output.metadata["attempts"] = attempts
            return output

        logger.warning(
            "answerer.failed attempts=%s error=%s", self.max_attempts, last_error
        )
        raise ValueError(f"Answerer failed to produce a valid answer: {last_error}")


def build_answerer_prompt(
    request: AnswererRequest,
    retry_feedback: str | None = None,
    settings: AnswererSettings | None = None,
) -> Prompt:
    """Build the prompt for one per-iteration answer submission."""

    settings = settings or AnswererSettings.load()
    task_images = encoded_task_context_images(request.encoded_task)
    output_correction = (
        "\n\nOUTPUT CORRECTION:\n"
        + retry_feedback
        + " Return only `<video_id>, <timestamp_ms>`."
        if retry_feedback
        else ""
    )
    return Prompt(
        system_prompt=_with_rejection_guidance(
            render_prompt(
                ANSWERER_SYSTEM_PROMPT,
                request.task_type,
                request.dataset,
            )
        ),
        context_prompt=render_prompt(
            ANSWERER_CONTEXT_PROMPT,
            request.task_type,
            request.dataset,
            TARGET=_encoded_task_text(request),
            SEARCH_EVIDENCE=_json(request.history),
            OUTPUT_CORRECTION=output_correction,
        ),
        max_tokens=min(settings.max_tokens, 64),
        task_images=task_images,
        task_videos=request.task_videos,
        context_images=[],
        context_videos=[],
        context=PromptContext(
            llm_agent=LlmAgent.ANSWERER,
            dataset=request.dataset,
            task_type=request.task_type,
        ),
    )


def _with_rejection_guidance(value: str) -> str:
    """Explain read-only retrieval rejection evidence to the answerer."""

    guidance = (
        "- `rejected_candidates` contains anonymized retrieval-loop submissions that "
        "an external checker rejected as the exact target. Treat the candidate and "
        "its recorded scene interval as strong negative evidence; the rejection does "
        "not imply that another timestamp in the same video is correct. This evidence "
        "never includes your own previous estimates or their validation.\n"
    )
    return value.replace("- Return exactly", guidance + "- Return exactly", 1)


def parse_answerer_output(response: str) -> AnswererOutput:
    """Parse the answerer's deliberately strict output format."""

    stripped = response.strip()
    match = _ANSWER_PATTERN.fullmatch(stripped)
    if match is None:
        raise ValueError(
            "Answer must contain exactly one video id, a comma, and one integer "
            "timestamp in milliseconds."
        )
    return AnswererOutput(
        video_id=match.group(1),
        timestamp_ms=int(match.group(2)),
    )


def _encoded_task_text(request: AnswererRequest) -> str:
    if request.encoded_task.text and request.encoded_task.text.strip():
        return remove_task_question(request.encoded_task.text)
    return _json(
        request.encoded_task.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"metadata", "notes", "status"},
        )
    )


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
