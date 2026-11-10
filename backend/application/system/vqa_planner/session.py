"""Execute one video-locked VQA-planner session after a browser match."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from application.system.qa import QuestionAnswerer
from application.system.timing import elapsed_ms, timer_start
from application.system.vqa import VQA
from application.system.vqa_planner.planner import VQAPlanner
from application.tasks.models import Dataset, EncodedTask, TaskCreate
from application.types.answerer.models import (
    AnswererValidation,
    AnswererValidationStatus,
)
from application.types.qa.models import QARequest
from application.types.retrieval.candidate import Candidate
from application.types.vlm.prompt import VideoInput
from application.types.vqa import VQAResult
from application.types.vqa_planner import (
    VQAPlannerAnswer,
    VQAPlannerDecision,
    VQAPlannerEndWithoutAnswer,
    VQAPlannerHistoryEntry,
    VQAPlannerInspectAllScenes,
    VQAPlannerInspectScene,
    VQAPlannerInspectScenes,
    VQAPlannerQueryKeyframes,
    VQAPlannerRequest,
    VQAPlannerScanFromStart,
    VQAPlannerSessionResult,
    VQAPlannerStep,
)


KeyframeRetriever = Callable[
    [str, str, TaskCreate, int],
    Awaitable[list[Candidate]],
]
AnswerValidator = Callable[[str], AnswererValidation]
QAVideoResolver = Callable[[str, int, int, Dataset], VideoInput]


class VQAPlannerSession:
    """Run agent-selected QA operations without leaving one matched video."""

    def __init__(
        self,
        *,
        planner: VQAPlanner,
        qa: QuestionAnswerer,
        vqa: VQA,
        video_resolver: QAVideoResolver,
        keyframe_retriever: KeyframeRetriever | None = None,
        candidate_count: int = 16,
        max_steps: int = 16,
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("VQA planner candidate_count must be greater than 0.")
        if max_steps <= 0:
            raise ValueError("VQA planner max_steps must be greater than 0.")
        self.planner = planner
        self.qa = qa
        self.vqa = vqa
        self.video_resolver = video_resolver
        self.keyframe_retriever = keyframe_retriever or _default_keyframe_retriever
        self.candidate_count = candidate_count
        self.max_steps = max_steps

    async def run(
        self,
        *,
        task: TaskCreate,
        encoded_task: EncodedTask,
        matched_candidate: Candidate,
        question: str,
        validate_answer: AnswerValidator,
        matched_scene_description: str | None = None,
    ) -> VQAPlannerSessionResult:
        source_video_id = _candidate_video_id(matched_candidate)
        invocation_timestamp_ms = matched_candidate.time_keyframe
        session_context = {
            "source_video_id": source_video_id,
            "invocation_timestamp_ms": invocation_timestamp_ms,
            "matched_scene_description": matched_scene_description,
        }
        if source_video_id is None:
            return VQAPlannerSessionResult(
                ended_without_answer=True,
                reason="The matched candidate has no source video id.",
                metadata=session_context,
            )

        question = question.strip()
        if not question:
            return VQAPlannerSessionResult(
                ended_without_answer=True,
                reason="The QA task has no question for the VQA planner.",
                metadata=session_context,
            )

        question_task = _question_only_task(task, question)
        history: list[VQAPlannerHistoryEntry] = []
        steps: list[VQAPlannerStep] = []
        candidates_by_id: dict[str, Candidate] = {}
        session_started = timer_start()

        for step_index in range(1, self.max_steps + 1):
            try:
                decision = await self.planner.plan(
                    VQAPlannerRequest(
                        question=question,
                        invocation_timestamp_ms=invocation_timestamp_ms,
                        matched_scene_description=matched_scene_description,
                        history=list(history),
                    )
                )
            except Exception as error:
                return VQAPlannerSessionResult(
                    ended_without_answer=True,
                    reason=f"The VQA planner could not choose an action: {error}",
                    steps=steps,
                    metadata={
                        **session_context,
                        "duration_ms": elapsed_ms(session_started),
                        "planner_error": str(error),
                    },
                )
            action = decision.action
            step_started = timer_start()

            if isinstance(action, VQAPlannerAnswer):
                validation = validate_answer(action.answer)
                accepted = validation.status == AnswererValidationStatus.ACCEPTED
                result = {
                    "accepted": accepted,
                    "validation": _planner_validation_feedback(validation),
                }
                step = VQAPlannerStep(
                    index=step_index,
                    decision=decision,
                    result=result,
                    validation=validation,
                    metadata={"duration_ms": elapsed_ms(step_started)},
                )
                steps.append(step)
                if accepted:
                    return VQAPlannerSessionResult(
                        answer=action.answer,
                        steps=steps,
                        metadata={
                            **session_context,
                            "duration_ms": elapsed_ms(session_started),
                        },
                    )
                history.append(_history_entry(decision, result))
                continue

            if isinstance(action, VQAPlannerEndWithoutAnswer):
                step = VQAPlannerStep(
                    index=step_index,
                    decision=decision,
                    result={"ended_without_answer": True, "reason": action.reason},
                    metadata={"duration_ms": elapsed_ms(step_started)},
                )
                steps.append(step)
                return VQAPlannerSessionResult(
                    ended_without_answer=True,
                    reason=action.reason,
                    steps=steps,
                    metadata={
                        **session_context,
                        "duration_ms": elapsed_ms(session_started),
                    },
                )

            try:
                result, operation_metadata = await self._execute_evidence_action(
                    decision=decision,
                    task=question_task,
                    encoded_task=encoded_task,
                    matched_candidate=matched_candidate,
                    source_video_id=source_video_id,
                    question=question,
                    candidates_by_id=candidates_by_id,
                )
            except Exception as error:
                result = {
                    "success": False,
                    "error": str(error),
                    "instruction": "Choose another valid action using known evidence.",
                }
                operation_metadata = {"error": str(error)}

            operation_metadata["duration_ms"] = elapsed_ms(step_started)
            step = VQAPlannerStep(
                index=step_index,
                decision=decision,
                result=result,
                metadata=operation_metadata,
            )
            steps.append(step)
            history.append(_history_entry(decision, result))

        return VQAPlannerSessionResult(
            ended_without_answer=True,
            reason=(
                f"The VQA planner exhausted its {self.max_steps} action steps without "
                "an accepted answer."
            ),
            steps=steps,
            metadata={
                **session_context,
                "duration_ms": elapsed_ms(session_started),
                "max_steps_exhausted": True,
            },
        )

    async def _execute_evidence_action(
        self,
        *,
        decision: VQAPlannerDecision,
        task: TaskCreate,
        encoded_task: EncodedTask,
        matched_candidate: Candidate,
        source_video_id: str,
        question: str,
        candidates_by_id: dict[str, Candidate],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        action = decision.action
        if isinstance(action, VQAPlannerQueryKeyframes):
            candidates = await self.keyframe_retriever(
                source_video_id,
                action.query,
                task,
                self.candidate_count,
            )
            locked_candidates = [
                candidate
                for candidate in candidates
                if _candidate_video_id(candidate) == source_video_id
            ]
            for candidate in locked_candidates:
                candidates_by_id[_candidate_id(candidate)] = candidate
            return (
                {
                    "success": True,
                    "query": action.query,
                    "keyframes": [
                        _keyframe_result(candidate) for candidate in locked_candidates
                    ],
                },
                {
                    "returned_candidate_count": len(candidates),
                    "locked_candidate_count": len(locked_candidates),
                },
            )

        if isinstance(action, VQAPlannerInspectScene):
            candidate = _known_keyframe_candidate(
                action.keyframe_id,
                candidates_by_id,
                source_video_id,
            )
            return await self._inspect_candidate_scene(
                candidate=candidate,
                source_video_id=source_video_id,
                question=question,
                task=task,
            )

        if isinstance(action, VQAPlannerInspectScenes):
            if len(set(action.keyframe_ids)) != len(action.keyframe_ids):
                raise ValueError("inspect_scenes keyframe_ids must be unique.")
            candidates = [
                _known_keyframe_candidate(
                    keyframe_id,
                    candidates_by_id,
                    source_video_id,
                )
                for keyframe_id in action.keyframe_ids
            ]
            inspections = await asyncio.gather(
                *(
                    self._inspect_candidate_scene(
                        candidate=candidate,
                        source_video_id=source_video_id,
                        question=question,
                        task=task,
                    )
                    for candidate in candidates
                )
            )
            return (
                {
                    "success": True,
                    "scenes": [result for result, _ in inspections],
                },
                _combined_scene_metadata(inspections),
            )

        if isinstance(action, VQAPlannerInspectAllScenes):
            result = await self._run_full_video_strategy(
                task=task,
                encoded_task=encoded_task,
                matched_candidate=matched_candidate,
                temporal_mode="V",
                start_timestamp_ms=action.start_timestamp_ms,
                end_timestamp_ms=action.end_timestamp_ms,
            )
            return _vqa_evidence(result), result.metadata

        if isinstance(action, VQAPlannerScanFromStart):
            result = await self._run_full_video_strategy(
                task=task,
                encoded_task=encoded_task,
                matched_candidate=matched_candidate,
                temporal_mode="T",
            )
            return _vqa_evidence(result), result.metadata

        raise TypeError(f"Unsupported VQA planner evidence action: {action.action}")

    async def _inspect_candidate_scene(
        self,
        *,
        candidate: Candidate,
        source_video_id: str,
        question: str,
        task: TaskCreate,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        video = self.video_resolver(
            source_video_id,
            candidate.time_scene_start,
            candidate.time_scene_end,
            task.dataset,
        )
        output = await self.qa.answer(
            QARequest(
                question=question,
                video=video,
                dataset=task.dataset,
                task_type=task.task_type,
            )
        )
        return (
            {
                "success": True,
                "keyframe_id": _candidate_id(candidate),
                "timestamp_ms": candidate.time_keyframe,
                "scene_start_ms": candidate.time_scene_start,
                "scene_end_ms": candidate.time_scene_end,
                "observation": output.answer,
            },
            output.metadata,
        )

    async def _run_full_video_strategy(
        self,
        *,
        task: TaskCreate,
        encoded_task: EncodedTask,
        matched_candidate: Candidate,
        temporal_mode: str,
        start_timestamp_ms: int | str = "START",
        end_timestamp_ms: int | str = "END",
    ) -> VQAResult:
        forced_encoded_task = encoded_task.model_copy(
            update={
                "text": None,
                "media_refs": [],
                "metadata": {
                    **encoded_task.metadata,
                    "qa_temporal_mode": temporal_mode,
                    "qa_time_specific": temporal_mode == "T",
                    "qa_inspection_start_ms": (
                        0 if start_timestamp_ms == "START" else start_timestamp_ms
                    ),
                    "qa_inspection_end_ms": (
                        None if end_timestamp_ms == "END" else end_timestamp_ms
                    ),
                },
            },
            deep=True,
        )
        return await self.vqa.answer(
            task,
            matched_candidate,
            encoded_task=forced_encoded_task,
            task_text=None,
            include_task_text=False,
            task_images=[],
            task_videos=[],
        )


async def _default_keyframe_retriever(
    video_id: str,
    query_text: str,
    task: TaskCreate,
    candidate_count: int,
) -> list[Candidate]:
    from application.system.retrieval.endpoints import get_endpoint_for_query
    from application.system.retrieval.fetch import (
        create_payload,
        fetch_candidates_for_query,
    )
    from application.system.retrieval_loop.retrieval_adapter import (
        retrieval_dataset_name,
    )
    from application.types.retrieval.queries import IntraVideoQuery

    query = IntraVideoQuery(
        dataset=retrieval_dataset_name(task.dataset.value),
        videoId=video_id,
        query=query_text,
        candidateCount=candidate_count,
    )
    endpoint = get_endpoint_for_query(query)
    payload = create_payload(query)
    return await fetch_candidates_for_query(query, endpoint, payload)


def _question_only_task(task: TaskCreate, question: str) -> TaskCreate:
    return task.model_copy(
        update={
            "input": task.input.model_copy(
                update={
                    "description": question,
                    "media": None,
                    "task_video": None,
                    "answer_text": None,
                    "answer_video": None,
                }
            )
        },
        deep=True,
    )


def _history_entry(
    decision: VQAPlannerDecision,
    result: dict[str, Any],
) -> VQAPlannerHistoryEntry:
    return VQAPlannerHistoryEntry(
        action=decision.action.model_dump(mode="json"),
        result=result,
    )


def _known_keyframe_candidate(
    keyframe_id: str,
    candidates_by_id: dict[str, Candidate],
    source_video_id: str,
) -> Candidate:
    candidate = candidates_by_id.get(keyframe_id)
    if candidate is None:
        raise ValueError(
            f"Unknown keyframe id {keyframe_id!r}; run query_keyframes and use "
            "one of its returned ids."
        )
    if _candidate_video_id(candidate) != source_video_id:
        raise ValueError("The requested keyframe is outside the matched video.")
    return candidate


def _combined_scene_metadata(
    inspections: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    scene_metadata: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []
    for result, metadata in inspections:
        keyframe_id = result["keyframe_id"]
        scene_metadata.append({"keyframe_id": keyframe_id, **metadata})
        calls = metadata.get("llm_calls")
        if isinstance(calls, list):
            llm_calls.extend(
                {**call, "keyframe_id": keyframe_id}
                for call in calls
                if isinstance(call, dict)
            )
    return {
        "scene_count": len(inspections),
        "scene_metadata": scene_metadata,
        "llm_calls": llm_calls,
    }


def _candidate_video_id(candidate: Candidate) -> str | None:
    if len(candidate.id) > 1:
        return "_".join(candidate.id[:-1])
    if candidate.id:
        return candidate.id[0]
    return None


def _keyframe_result(candidate: Candidate) -> dict[str, Any]:
    return {
        "keyframe_id": _candidate_id(candidate),
        "rank": candidate.rank,
        "score": candidate.score,
        "timestamp_ms": candidate.time_keyframe,
        "scene_start_ms": candidate.time_scene_start,
        "scene_end_ms": candidate.time_scene_end,
    }


def _candidate_id(candidate: Candidate) -> str:
    return "_".join(candidate.id)


def _vqa_evidence(result: VQAResult) -> dict[str, Any]:
    return {
        "success": True,
        "answerable": result.answerable,
        "proposed_answer": result.answer,
        "reason": result.reason,
        "scene_observations": [
            {
                "segment_index": observation.index,
                "start_seconds": observation.start_seconds,
                "end_seconds": observation.end_seconds,
                "observation": observation.answer,
                "verified": observation.verified,
                "verification_reason": observation.verification_reason,
            }
            for observation in result.observations
        ],
    }


def _planner_validation_feedback(
    validation: AnswererValidation,
) -> dict[str, Any]:
    if validation.status == AnswererValidationStatus.ACCEPTED:
        reason = "The submitted answer was accepted by the outside answer checker."
    elif validation.status == AnswererValidationStatus.REJECTED:
        reason = "The submitted answer was rejected by the outside answer checker."
    else:
        reason = (
            "The submitted answer could not be validated and was not accepted by "
            "the outside answer checker."
        )
    return {
        "status": validation.status.value,
        "reason": reason,
        "source": validation.source,
    }
