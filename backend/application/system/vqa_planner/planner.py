"""Decision boundary and prompt-backed VQA planning agent."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import TypeAdapter, ValidationError

from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.executor import run
from application.system.vlm.prompt_trace import prompt_trace
from application.system.vqa_planner.prompt import VQA_PLANNER_SYSTEM_PROMPT
from application.types.vlm.prompt import LlmAgent, Prompt, PromptContext
from application.types.vqa_planner import (
    VQAPlannerAction,
    VQAPlannerDecision,
    VQAPlannerRequest,
)


VlmRunner = Callable[[str, Prompt], Awaitable[str]]
_ACTION_ADAPTER = TypeAdapter(VQAPlannerAction)


class VQAPlanner(ABC):
    """Choose the next evidence-gathering action inside one matched video."""

    @abstractmethod
    async def plan(self, request: VQAPlannerRequest) -> VQAPlannerDecision:
        """Return one executable video-locked QA action."""


class PromptVQAPlanner(VQAPlanner):
    def __init__(
        self,
        model: str = "qwen",
        runner: VlmRunner = run,
        *,
        max_tokens: int = 512,
        max_attempts: int = 3,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("VQA planner max_tokens must be greater than 0.")
        if max_attempts <= 0:
            raise ValueError("VQA planner max_attempts must be greater than 0.")
        self.model = model
        self.runner = runner
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts

    async def plan(self, request: VQAPlannerRequest) -> VQAPlannerDecision:
        calls: list[dict[str, Any]] = []
        retry_feedback: str | None = None
        last_error: ValueError | None = None
        for attempt in range(1, self.max_attempts + 1):
            prompt = build_vqa_planner_prompt(
                request,
                max_tokens=self.max_tokens,
                retry_feedback=retry_feedback,
            )
            started = timer_start()
            response = await self.runner(self.model, prompt)
            duration_ms = elapsed_ms(started)
            call = {
                "prompt": prompt_trace(prompt),
                "response": response,
                "model": self.model,
                "attempt": attempt,
                "duration_ms": duration_ms,
            }
            try:
                action = parse_vqa_planner_action(response)
            except ValueError as error:
                call["error"] = str(error)
                calls.append(call)
                retry_feedback = str(error)
                last_error = error
                continue
            calls.append(call)
            return VQAPlannerDecision(
                action=action,
                metadata={"llm_calls": calls},
            )

        raise ValueError(
            "VQA planner failed to produce a valid action after "
            f"{self.max_attempts} attempts: {last_error}"
        )


def build_vqa_planner_prompt(
    request: VQAPlannerRequest,
    *,
    max_tokens: int = 512,
    retry_feedback: str | None = None,
) -> Prompt:
    """Build a prompt with the explicit task, matched scene, and action history."""

    history: Any = [entry.model_dump(mode="json") for entry in request.history]
    return Prompt(
        system_prompt=VQA_PLANNER_SYSTEM_PROMPT,
        context_prompt=json.dumps(
            {
                "task_to_solve": {"question": request.question},
                "matched_scene_context": {
                    "description": request.matched_scene_description,
                    "invocation_timestamp_ms": request.invocation_timestamp_ms,
                },
                "action_history": history,
                **(
                    {"output_correction": retry_feedback}
                    if retry_feedback is not None
                    else {}
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        temperature=0,
        max_tokens=max_tokens,
        task_text=None,
        task_images=[],
        task_videos=[],
        context_images=[],
        context_videos=[],
        context=PromptContext(llm_agent=LlmAgent.VQA_PLANNER),
    )


def parse_vqa_planner_action(response: str) -> VQAPlannerAction:
    value = _strip_code_fence(response)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("VQA planner must return one valid JSON object.") from error
    try:
        return _ACTION_ADAPTER.validate_python(payload)
    except ValidationError as error:
        raise ValueError(f"Invalid VQA planner action: {error}") from error


def _strip_code_fence(response: str) -> str:
    lines = response.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
