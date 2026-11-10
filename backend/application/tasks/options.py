"""Task and dataset metadata shared by API clients."""

from __future__ import annotations

from application.tasks.models import (
    Dataset,
    DatasetOption,
    TaskOption,
    TaskOptionsResponse,
    TaskType,
)


TASK_OPTIONS = [
    TaskOption(
        value=TaskType.VISUAL_KIS,
        label="Visual KIS",
        required_inputs=["media"],
        description="Find the exact scene specified by visual media or a video segment.",
    ),
    TaskOption(
        value=TaskType.TEXTUAL_KIS,
        label="Textual KIS",
        required_inputs=["text"],
        description="Find the exact scene specified by a textual description.",
    ),
    TaskOption(
        value=TaskType.QA,
        label="Question Answering",
        required_inputs=["text", "media"],
        description="Answer a question, optionally grounded in supplied visual context.",
    ),
    TaskOption(
        value=TaskType.AVS,
        label="Ad-hoc Video Search",
        required_inputs=["text"],
        description="Find as many scenes as possible matching the task description.",
    ),
]

DATASET_OPTIONS = [
    DatasetOption(
        value=Dataset.MVK,
        label="MVK",
        description="Multimedia retrieval dataset of underwater life.",
    ),
    DatasetOption(
        value=Dataset.V3C,
        label="V3C",
        description="V3C video collection.",
    ),
    DatasetOption(
        value=Dataset.LHE,
        label="LHE",
        description="LSC-style lifelog dataset.",
    ),
]


def get_task_options() -> TaskOptionsResponse:
    """Return task setup metadata for API clients."""

    return TaskOptionsResponse(task_types=TASK_OPTIONS, datasets=DATASET_OPTIONS)

