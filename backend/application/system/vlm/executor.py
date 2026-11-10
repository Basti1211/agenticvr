from application.system.vlm.anonymize import (
    MediaIdAnonymizer,
    anonymize_prompt_media_ids,
    anonymize_prompt_media_ids_batch,
)
from application.system.vlm.candidate_video_aliases import (
    anonymize_candidate_video_ids,
    deanonymize_candidate_video_ids,
)
from application.system.vlm.instance.qwen.model import (
    run_qwen,
    run_qwen_parallel,
)
from application.types.vlm.prompt import BatchedPrompt, LlmAgent, Prompt


CANDIDATE_ID_VISIBLE_AGENTS = {
    LlmAgent.ANSWERER,
    LlmAgent.BROWSER_CLASSIFIER,
    LlmAgent.SECOND_STAGE_EVALUATOR,
}


async def run(model: str, prompt: Prompt) -> str:
    prompt = _prompt_for_vlm(prompt)
    prompt = anonymize_candidate_video_ids(prompt)
    if model == "qwen":
        response = await run_qwen(prompt)
        return deanonymize_candidate_video_ids(response)

    return "Test"


async def run_parallel(model: str, prompts: BatchedPrompt | list[Prompt]) -> list[str]:
    if isinstance(prompts, list) and not prompts:
        return []

    batched_prompts = (
        prompts
        if isinstance(prompts, BatchedPrompt)
        else BatchedPrompt(prompts=prompts)
    )
    batched_prompts = BatchedPrompt(
        prompts=[
            anonymize_candidate_video_ids(prompt)
            for prompt in _prompts_for_vlm(batched_prompts.prompts)
        ]
    )
    if model == "qwen":
        responses = await run_qwen_parallel(batched_prompts)
        return [deanonymize_candidate_video_ids(response) for response in responses]
    return ["Test" for _ in batched_prompts.prompts]


def _prompt_for_vlm(prompt: Prompt) -> Prompt:
    if _should_preserve_media_ids(prompt):
        return prompt
    return anonymize_prompt_media_ids(prompt)


def _prompts_for_vlm(prompts: list[Prompt]) -> list[Prompt]:
    if all(_should_preserve_media_ids(prompt) for prompt in prompts):
        return prompts
    if not any(_should_preserve_media_ids(prompt) for prompt in prompts):
        return anonymize_prompt_media_ids_batch(prompts)

    anonymizer = MediaIdAnonymizer()
    return [
        prompt
        if _should_preserve_media_ids(prompt)
        else anonymizer.anonymize_prompt(prompt)
        for prompt in prompts
    ]


def _should_preserve_media_ids(prompt: Prompt) -> bool:
    return (
        prompt.context is not None
        and prompt.context.llm_agent in CANDIDATE_ID_VISIBLE_AGENTS
    )
