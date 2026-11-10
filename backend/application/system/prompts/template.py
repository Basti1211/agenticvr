"""Strict selection and rendering for task/dataset prompt templates."""

from __future__ import annotations

from string import Template
from typing import TypeAlias

from application.tasks.models import Dataset, TaskType


PromptKey: TypeAlias = tuple[TaskType, Dataset]
PromptTemplates: TypeAlias = dict[PromptKey, str]


def render_prompt(
    prompts: PromptTemplates,
    task_type: TaskType | str | None,
    dataset: Dataset | str | None,
    /,
    **context: object,
) -> str:
    """Select one exact prompt variant and strictly replace its placeholders."""

    if task_type is None or dataset is None:
        raise ValueError("Prompt rendering requires both task_type and dataset.")

    try:
        prompt_key = (TaskType(task_type), Dataset(dataset))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid prompt task/dataset combination: {task_type!r}, {dataset!r}."
        ) from error

    try:
        prompt = prompts[prompt_key]
    except KeyError as error:
        raise ValueError(
            "No prompt configured for "
            f"task_type={prompt_key[0].value!r}, dataset={prompt_key[1].value!r}."
        ) from error

    return Template(prompt).substitute(
        {name: str(value) for name, value in context.items()}
    )
