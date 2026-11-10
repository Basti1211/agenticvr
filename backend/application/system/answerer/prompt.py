"""Answerer prompt variants for every task and dataset combination."""

from application.system.prompts.template import PromptTemplates
from application.tasks.models import Dataset, TaskType


ANSWERER_SYSTEM_PROMPT: PromptTemplates = {
    (
        TaskType.VISUAL_KIS,
        Dataset.MVK,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.VISUAL_KIS,
        Dataset.V3C,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.VISUAL_KIS,
        Dataset.LHE,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.TEXTUAL_KIS,
        Dataset.MVK,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.TEXTUAL_KIS,
        Dataset.V3C,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.TEXTUAL_KIS,
        Dataset.LHE,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.QA,
        Dataset.MVK,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.QA,
        Dataset.V3C,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.QA,
        Dataset.LHE,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.AVS,
        Dataset.MVK,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.AVS,
        Dataset.V3C,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
    (
        TaskType.AVS,
        Dataset.LHE,
    ): """Submit the single most likely video answer using the available search evidence.
Rules:
- Choose the video and timestamp that appear most likely.
- The timestamp must be a non-negative integer in milliseconds.
- Return exactly `<video_id>, <timestamp_ms>`.
- Return no prefix, JSON, markdown, label, or explanation.""",
}


ANSWERER_CONTEXT_PROMPT: PromptTemplates = {
    (TaskType.VISUAL_KIS, Dataset.MVK): """Specific context:
- Submit the exact video segment represented by the visual target. Close duplicates are not acceptable.
- The candidate videos come from a highly homogeneous marine collection. Prefer evidence with the exact species, colors, shapes, and background.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.VISUAL_KIS, Dataset.V3C): """Specific context:
- Submit the exact video segment represented by the visual target. Close duplicates are not acceptable.
- The target scene is from a collection of everyday-life videos.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.VISUAL_KIS, Dataset.LHE): """Specific context:
- Submit the exact video segment represented by the visual target. Close duplicates are not acceptable.
- The candidate videos come from a highly homogeneous laparoscopic collection. Prefer exact tissue patterns, colors, shapes, instruments, and actions.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.TEXTUAL_KIS, Dataset.MVK): """Specific context:
- Submit the exact scene or segment described by the task. Close duplicates are not acceptable.
- The candidate videos come from a highly homogeneous marine collection. Prefer evidence with the exact species, colors, shapes, and background.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.TEXTUAL_KIS, Dataset.V3C): """Specific context:
- Submit the exact scene or segment described by the task. Close duplicates are not acceptable.
- The target scene is from a collection of everyday-life videos.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.TEXTUAL_KIS, Dataset.LHE): """Specific context:
- Submit the exact scene or segment described by the task. Close duplicates are not acceptable.
- The candidate videos come from a highly homogeneous laparoscopic collection. Prefer exact tissue patterns, colors, shapes, instruments, and actions.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.QA, Dataset.MVK): """Specific context:
- Submit the video segment that best grounds the requested answer in the available evidence.
- The candidate videos come from a highly homogeneous marine collection. Prefer evidence with the exact species, colors, shapes, and background.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.QA, Dataset.V3C): """Specific context:
- Submit the video segment that best grounds the requested answer in the available evidence.
- The target scene is from a collection of everyday-life videos.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.QA, Dataset.LHE): """Specific context:
- Submit the video segment that best grounds the requested answer in the available evidence.
- The candidate videos come from a highly homogeneous laparoscopic collection. Prefer exact tissue patterns, colors, shapes, instruments, and actions.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.AVS, Dataset.MVK): """Specific context:
- Submit the video segment that most strongly satisfies all requested ad-hoc search criteria.
- The candidate videos come from a highly homogeneous marine collection. Prefer evidence with the exact species, colors, shapes, and background.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.AVS, Dataset.V3C): """Specific context:
- Submit the video segment that most strongly satisfies all requested ad-hoc search criteria.
- The target scene is from a collection of everyday-life videos.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
    (TaskType.AVS, Dataset.LHE): """Specific context:
- Submit the video segment that most strongly satisfies all requested ad-hoc search criteria.
- The candidate videos come from a highly homogeneous laparoscopic collection. Prefer exact tissue patterns, colors, shapes, instruments, and actions.

TARGET:
$TARGET

SEARCH EVIDENCE:
$SEARCH_EVIDENCE$OUTPUT_CORRECTION

ANSWER:""",
}
