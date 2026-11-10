"""VQA prompt variants for every concrete question answering agent."""

from application.system.prompts.template import PromptTemplates
from application.tasks.models import Dataset, TaskType


_DATASET_CONTEXT = {
    Dataset.MVK: (
        "The video is from a collection of Marine Videos that are highly "
        "homogeneous. Focus on fine-grained visual differences, such as "
        "distinctive species, background colors, and shapes."
    ),
    Dataset.V3C: (
        "The video is from a collection of everyday life videos. Encode scene "
        "category, visible objects, actions, people, text, temporal changes, "
        "and setting."
    ),
    Dataset.LHE: (
        "The video is from a collection of laproscopic videos that are highly "
        "homogeneous. Focus on fine-grained visual differences, tissue "
        "patterns, background colors, and shapes."
    ),
}


def _segment_answer_context(dataset_context: str) -> str:
    return f"""Specific context:
- Your goal is to answer the original question using only the attached video segment.
- {dataset_context}

QUESTION:
$QUESTION

VQA ITERATION:
$VQA_ITERATION

SEGMENT:
- global_index: $SEGMENT_INDEX
- segments_in_current_iteration: $ITERATION_SEGMENT_COUNT
- source_candidate_id: $SOURCE_CANDIDATE_ID
- segment_start_seconds: $START_SECONDS
- segment_end_seconds: $END_SECONDS
$ANSWER_RETRY_CONTEXT

Answer only from this attached segment."""


def _answer_verification_context(dataset_context: str) -> str:
    return f"""Specific context:
- Your goal is to verify whether the proposed answer is directly supported by the attached video segment.
- {dataset_context}

QUESTION:
$QUESTION

PROPOSED ANSWER:
$PROPOSED_ANSWER

SEGMENT:
- global_index: $SEGMENT_INDEX
- source_candidate_id: $SOURCE_CANDIDATE_ID
- segment_start_seconds: $START_SECONDS
- segment_end_seconds: $END_SECONDS

Verify only against this attached segment."""


def _answer_selection_context(dataset_context: str) -> str:
    return f"""Specific context:
- Your goal is to choose the best answer from multiple verified answer options.
- {dataset_context}

QUESTION:
$QUESTION

VERIFIED ANSWERS:
$VERIFIED_ANSWERS

Return the global_index of the best answer."""


def _missing_observation_context(dataset_context: str) -> str:
    return f"""Specific context:
- Your goal is to inspect why this attached segment did not answer the original question.
- {dataset_context}

QUESTION:
$QUESTION

SEGMENT:
- global_index: $SEGMENT_INDEX
- source_candidate_id: $SOURCE_CANDIDATE_ID
- segment_start_seconds: $START_SECONDS
- segment_end_seconds: $END_SECONDS

SEGMENT ANSWER ATTEMPT:
$SEGMENT_ANSWER_ATTEMPT

VERIFICATION REASON:
$VERIFICATION_REASON

Describe only visible or audible evidence and missing information."""


def _missing_reasoning_context(dataset_context: str) -> str:
    return f"""Specific context:
- Your goal is to explain why the inspected video evidence could not answer the original question.
- {dataset_context}

QUESTION:
$QUESTION

SEGMENT EVIDENCE:
$SEGMENT_EVIDENCE

Use only the supplied segment evidence."""


def _scene_notes_context(dataset_context: str) -> str:
    return f"""Specific context:
- Inspect this attached video segment specifically for the exact evidence needed to answer the original question after other segments are also reviewed. Preserve exact names, numbers, visible text, and other answer-bearing details. Omit generic scene description that does not help answer the question.
- {dataset_context}

QUESTION:
$QUESTION

SEGMENT:
- global_index: $SEGMENT_INDEX
- segments_in_current_iteration: $ITERATION_SEGMENT_COUNT
- source_candidate_id: $SOURCE_CANDIDATE_ID
- segment_start_seconds: $START_SECONDS
- segment_end_seconds: $END_SECONDS
$ANSWER_RETRY_CONTEXT

Write only notes grounded in this attached segment."""


def _scene_notes_answer_context(dataset_context: str) -> str:
    return f"""Specific context:
- Precisely answer the original question using the chronological notes collected from video scenes. Return the exact type and granularity requested by the question, such as a city name, count, person, object, or action.
- {dataset_context}

QUESTION:
$QUESTION

CHRONOLOGICAL SCENE NOTES:
$SCENE_NOTES
$ANSWER_RETRY_CONTEXT

Use only the supplied scene notes."""


SEGMENT_ANSWER_SYSTEM_PROMPT: PromptTemplates = {
    (
        TaskType.QA,
        Dataset.MVK,
    ): """Decide whether the question can be answered using only the attached video segment. Inspect the entire segment, including visible actions, objects, text, temporal changes, and audible information when available. Do not use outside knowledge. Answer only if the segment contains enough evidence to answer the question. If the answer-bearing text or detail is present but hard to read, make a cautious best-effort reading only when the visible evidence strongly narrows the answer; do not invent unsupported text or make a total guess.

Return exactly one line:
A: <direct answer>
or
F

Return F when there is no definite answer available in this segment alone or the hard-to-read evidence remains too ambiguous. Do not explain your reasoning or provide any additional information.""",
    (
        TaskType.QA,
        Dataset.V3C,
    ): """Decide whether the question can be answered using only the attached video segment. Inspect the entire segment, including visible actions, objects, text, temporal changes, and audible information when available. Do not use outside knowledge. Answer only if the segment contains enough evidence to answer the question. If the answer-bearing text or detail is present but hard to read, make a cautious best-effort reading only when the visible evidence strongly narrows the answer; do not invent unsupported text or make a total guess.

Return exactly one line:
A: <direct answer>
or
F

Return F when there is no definite answer available in this segment alone or the hard-to-read evidence remains too ambiguous. Do not explain your reasoning or provide any additional information.""",
    (
        TaskType.QA,
        Dataset.LHE,
    ): """Decide whether the question can be answered using only the attached video segment. Inspect the entire segment, including visible actions, objects, text, temporal changes, and audible information when available. Do not use outside knowledge. Answer only if the segment contains enough evidence to answer the question. If the answer-bearing text or detail is present but hard to read, make a cautious best-effort reading only when the visible evidence strongly narrows the answer; do not invent unsupported text or make a total guess.

Return exactly one line:
A: <direct answer>
or
F

Return F when there is no definite answer available in this segment alone or the hard-to-read evidence remains too ambiguous. Do not explain your reasoning or provide any additional information.""",
}


SEGMENT_ANSWER_CONTEXT_PROMPT: PromptTemplates = {
    (TaskType.QA, dataset): _segment_answer_context(dataset_context)
    for dataset, dataset_context in _DATASET_CONTEXT.items()
}


ANSWER_VERIFICATION_SYSTEM_PROMPT: PromptTemplates = {
    (
        TaskType.QA,
        Dataset.MVK,
    ): """Independently verify whether the proposed answer is fully supported by the attached source video segment. Inspect the segment yourself. Do not accept an answer merely because it sounds plausible. For hard-to-read answer-bearing text or details, accept only when the visible evidence strongly supports the proposed reading; reject when it remains too ambiguous.

Return exactly one line:
M: <short concrete supporting evidence>
or
F: <short concrete contradiction or missing evidence>

Use M only when the segment directly supports the complete answer.""",
    (
        TaskType.QA,
        Dataset.V3C,
    ): """Independently verify whether the proposed answer is fully supported by the attached source video segment. Inspect the segment yourself. Do not accept an answer merely because it sounds plausible. For hard-to-read answer-bearing text or details, accept only when the visible evidence strongly supports the proposed reading; reject when it remains too ambiguous.

Return exactly one line:
M: <short concrete supporting evidence>
or
F: <short concrete contradiction or missing evidence>

Use M only when the segment directly supports the complete answer.""",
    (
        TaskType.QA,
        Dataset.LHE,
    ): """Independently verify whether the proposed answer is fully supported by the attached source video segment. Inspect the segment yourself. Do not accept an answer merely because it sounds plausible. For hard-to-read answer-bearing text or details, accept only when the visible evidence strongly supports the proposed reading; reject when it remains too ambiguous.

Return exactly one line:
M: <short concrete supporting evidence>
or
F: <short concrete contradiction or missing evidence>

Use M only when the segment directly supports the complete answer.""",
}


ANSWER_VERIFICATION_CONTEXT_PROMPT: PromptTemplates = {
    (TaskType.QA, dataset): _answer_verification_context(dataset_context)
    for dataset, dataset_context in _DATASET_CONTEXT.items()
}


ANSWER_SELECTION_SYSTEM_PROMPT: PromptTemplates = {
    (
        TaskType.QA,
        Dataset.MVK,
    ): """Select the answer that most directly and completely answers the original question. Return only the integer segment index.""",
    (
        TaskType.QA,
        Dataset.V3C,
    ): """Select the answer that most directly and completely answers the original question. Return only the integer segment index.""",
    (
        TaskType.QA,
        Dataset.LHE,
    ): """Select the answer that most directly and completely answers the original question. Return only the integer segment index.""",
}


ANSWER_SELECTION_CONTEXT_PROMPT: PromptTemplates = {
    (TaskType.QA, dataset): _answer_selection_context(dataset_context)
    for dataset, dataset_context in _DATASET_CONTEXT.items()
}

MISSING_EVIDENCE_OBSERVATION_SYSTEM_PROMPT: PromptTemplates = {
    (
        TaskType.QA,
        Dataset.MVK,
    ): """Inspect the attached segment and explain only what information relevant to the question is visible or audible, and what required information is absent, hidden, ambiguous, or cut off. Do not answer the question and do not speculate.""",
    (
        TaskType.QA,
        Dataset.V3C,
    ): """Inspect the attached segment and explain only what information relevant to the question is visible or audible, and what required information is absent, hidden, ambiguous, or cut off. Do not answer the question and do not speculate.""",
    (
        TaskType.QA,
        Dataset.LHE,
    ): """Inspect the attached segment and explain only what information relevant to the question is visible or audible, and what required information is absent, hidden, ambiguous, or cut off. Do not answer the question and do not speculate.""",
}

MISSING_EVIDENCE_OBSERVATION_CONTEXT_PROMPT: PromptTemplates = {
    (TaskType.QA, dataset): _missing_observation_context(dataset_context)
    for dataset, dataset_context in _DATASET_CONTEXT.items()
}

MISSING_EVIDENCE_REASONING_SYSTEM_PROMPT: PromptTemplates = {
    (
        TaskType.QA,
        Dataset.MVK,
    ): """Explain why the complete video cannot answer the original question. Use only the supplied segment evidence. State the specific event, object, text, action, temporal relation, or audio information that is missing or remains ambiguous. Return exactly one concise sentence.""",
    (
        TaskType.QA,
        Dataset.V3C,
    ): """Explain why the complete video cannot answer the original question. Use only the supplied segment evidence. State the specific event, object, text, action, temporal relation, or audio information that is missing or remains ambiguous. Return exactly one concise sentence.""",
    (
        TaskType.QA,
        Dataset.LHE,
    ): """Explain why the complete video cannot answer the original question. Use only the supplied segment evidence. State the specific event, object, text, action, temporal relation, or audio information that is missing or remains ambiguous. Return exactly one concise sentence.""",
}

MISSING_EVIDENCE_REASONING_CONTEXT_PROMPT: PromptTemplates = {
    (TaskType.QA, dataset): _missing_reasoning_context(dataset_context)
    for dataset, dataset_context in _DATASET_CONTEXT.items()
}


SCENE_NOTES_SYSTEM_PROMPT: PromptTemplates = {
    (
        TaskType.QA,
        dataset,
    ): """Inspect the attached video segment specifically for evidence that can answer the supplied question. Record exact answer-bearing details, including names, numbers, visible text, actions, event order, and audible information when relevant. If answer-bearing text or detail is hard to read but visibly present, record a cautious likely reading only when the visual evidence strongly narrows it; do not invent unsupported text or make a total guess. If the direct answer is visible or audible, state it precisely. Do not add broad scene descriptions that do not help answer the question.

Return exactly one line:
S: <concise relevant scene notes>
or
N

Return N when this segment contains no useful information for answering the question or the hard-to-read evidence remains too ambiguous. Do not use outside knowledge."""
    for dataset in Dataset
}


SCENE_NOTES_CONTEXT_PROMPT: PromptTemplates = {
    (TaskType.QA, dataset): _scene_notes_context(dataset_context)
    for dataset, dataset_context in _DATASET_CONTEXT.items()
}


SCENE_NOTES_ANSWER_SYSTEM_PROMPT: PromptTemplates = {
    (
        TaskType.QA,
        dataset,
    ): """Precisely answer the question using only the supplied chronological scene notes. Combine notes across scenes when needed, and respect the scene order for questions about first, last, before, or after. Return the shortest exact answer that satisfies what the question asks; preserve exact names, numbers, and visible text, including cautious likely readings from hard-to-read evidence when the notes explicitly support them. Do not summarize the video.

Return exactly one line:
A: <direct answer>
or
F

Return F only when the supplied notes do not contain enough evidence."""
    for dataset in Dataset
}


SCENE_NOTES_ANSWER_CONTEXT_PROMPT: PromptTemplates = {
    (TaskType.QA, dataset): _scene_notes_answer_context(dataset_context)
    for dataset, dataset_context in _DATASET_CONTEXT.items()
}
