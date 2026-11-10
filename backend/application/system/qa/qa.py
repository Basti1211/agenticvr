"""Question-answering boundary and VLM-backed implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from application.settings.settings import _qa_settings
from application.system.prompts.template import render_prompt
from application.system.qa.prompt import QA_CONTEXT_PROMPT, QA_SYSTEM_PROMPT
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.executor import run
from application.system.vlm.prompt_trace import prompt_trace
from application.types.qa.models import QAOutput, QARequest
from application.types.vlm.prompt import LlmAgent, Prompt, PromptContext


VlmRunner = Callable[[str, Prompt], Awaitable[str]]


class QASettings:
    """Generation and video-materialization settings for QA."""

    def __init__(
        self,
        max_tokens: int,
        fps: int,
        resolution_width: int,
        resolution_height: int,
        crf: int = 23,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        if fps <= 0:
            raise ValueError("fps must be greater than 0.")
        if resolution_width <= 0 or resolution_height <= 0:
            raise ValueError("QA video resolution must be greater than 0.")
        if not 0 <= crf <= 51:
            raise ValueError("QA video CRF must be between 0 and 51.")

        self.max_tokens = max_tokens
        self.fps = fps
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.crf = crf

    @classmethod
    def load(cls) -> "QASettings":
        settings = _qa_settings()
        return cls(
            max_tokens=int(settings["max_tokens"]),
            fps=int(settings["fps"]),
            resolution_width=int(settings["resolution_width"]),
            resolution_height=int(settings["resolution_height"]),
            crf=int(settings.get("crf", 23)),
        )


class QuestionAnswerer(ABC):
    """Answers a textual question using one supplied media source."""

    @abstractmethod
    async def answer(self, request: QARequest) -> QAOutput:
        """Return an answer grounded in the request media."""


class PromptQuestionAnswerer(QuestionAnswerer):
    """Question answerer backed by the configured VLM."""

    def __init__(
        self,
        model: str = "qwen",
        runner: VlmRunner = run,
        settings: QASettings | None = None,
    ) -> None:
        self.model = model
        self.runner = runner
        self.settings = settings or QASettings.load()

    async def answer(self, request: QARequest) -> QAOutput:
        prompt = build_qa_prompt(request, self.settings)
        llm_start = timer_start()
        response = await self.runner(self.model, prompt)
        duration_ms = elapsed_ms(llm_start)
        answer = response.strip()
        if not answer:
            raise ValueError("QA model returned an empty answer.")

        return QAOutput(
            answer=answer,
            metadata={
                "llm_calls": [
                    {
                        "prompt": prompt_trace(prompt),
                        "response": response,
                        "model": self.model,
                        "duration_ms": duration_ms,
                    }
                ]
            },
        )


def build_qa_prompt(
    request: QARequest,
    settings: QASettings | None = None,
) -> Prompt:
    """Build a prompt that answers one question from one media source."""

    settings = settings or QASettings.load()
    return Prompt(
        system_prompt=render_prompt(
            QA_SYSTEM_PROMPT,
            request.task_type,
            request.dataset,
        ),
        context_prompt=render_prompt(
            QA_CONTEXT_PROMPT,
            request.task_type,
            request.dataset,
            QUESTION=json.dumps(request.question.strip(), ensure_ascii=False),
        ),
        max_tokens=settings.max_tokens,
        context_images=[request.image] if request.image is not None else [],
        context_videos=[request.video] if request.video is not None else [],
        context=PromptContext(
            llm_agent=LlmAgent.QA,
            dataset=request.dataset,
            task_type=request.task_type,
        ),
    )
