#!/usr/bin/env python3
"""Run imported tasks with one named experiment configuration."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypedDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from application.tasks.imported import (  # noqa: E402
    DEFAULT_IMPORTED_TASKS_DIR,
    imported_task_to_payload,
    load_downloaded_tasks,
    load_imported_tasks,
)
from application.tasks.logs import (  # noqa: E402
    list_task_run_logs,
    save_task_run_log,
    task_run_log_dir,
    task_run_log_path,
)
from application.tasks.models import (  # noqa: E402
    EncodedTask,
    ImportedTaskOption,
    TaskCreate,
    TaskType,
)
from application.settings.settings import (  # noqa: E402
    experiment_settings_dir,
    retrieval_candidate_count,
    use_experiment_settings,
)
from application.system.timing import add_timing_log_filter  # noqa: E402
from application.types.answerer.models import AnswererValidationStatus  # noqa: E402
from application.types.retrieval_loop.models import (  # noqa: E402
    RetrievalLoopIteration,
    RetrievalLoopResult,
    RetrievalLoopStatus,
)

FOUND_VIA_ANSWERER_STATUS = "found via answerer"
TASK_TIMEOUT_SECONDS = 350
T_KIS_TASK_TIMEOUT_SECONDS = 470
COMPLETED_TASK_STATUSES = {
    RetrievalLoopStatus.FOUND.value,
    RetrievalLoopStatus.NOT_FOUND.value,
    RetrievalLoopStatus.TIMEOUT.value,
    FOUND_VIA_ANSWERER_STATUS,
}
EXPERIMENTS_FILE_NAME = "experiments.json"
EXPERIMENT_TASK_TYPES = {"v-kis", "t-kis", "avs", "qa", "all"}
EXPERIMENT_DATASETS = {"lhe", "mvk", "v3c", "all"}
EXPERIMENT_HINT_INDICES = {0, 1, 2}
DEFAULT_HINT_INDEX = 2
ITERATIVE_HINT_SELECTION = "iterative"
HintSelection = int | Literal["iterative"]


class ExperimentSettings(TypedDict):
    task_type: str
    dataset: str
    max_iterations: int
    hint_index: HintSelection
    hide_avs_submission_ratings_from_planner: bool


def main() -> None:
    args = _parse_args()
    with use_experiment_settings(args.setting_name):
        _configure_logging()
        asyncio.run(run_imported_tasks(args))


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for handler in logging.getLogger().handlers:
        add_timing_log_filter(handler)
    logging.getLogger("application").setLevel(level)


async def run_imported_tasks(args: argparse.Namespace) -> None:
    # These modules contain settings-derived import-time defaults, so they must
    # be loaded only after main() selects the experiment settings directory.
    from application.system.retrieval_loop import default_retrieval_loop
    from application.tasks.service import task_service

    if args.tasks_file is None:
        tasks = load_downloaded_tasks()
        tasks_source = DEFAULT_IMPORTED_TASKS_DIR
    else:
        tasks = load_imported_tasks(args.tasks_file)
        tasks_source = args.tasks_file
    tasks = _filter_tasks(tasks, args, args.experiment_filters)
    tasks = _select_tkis_hints(tasks, args.experiment_filters["hint_index"])
    run_id = args.run_id or _default_run_id()
    log_dir = _task_log_dir(args.setting_name)
    tasks, skipped_tasks = _skip_completed_tasks(tasks, log_dir)
    max_candidates = (
        retrieval_candidate_count()
        if args.max_candidates is None
        else args.max_candidates
    )
    task_logs: list[dict[str, Any]] = []

    print(f"Batch run id: {run_id}")
    print(f"Experiment setting: {args.setting_name}")
    print(f"Writing one task log per imported task to {log_dir}")
    print(
        "Per-task runtime limits: "
        f"T-KIS={T_KIS_TASK_TIMEOUT_SECONDS} seconds, "
        f"all others={TASK_TIMEOUT_SECONDS} seconds"
    )
    print(f"T-KIS hint index: {args.experiment_filters['hint_index']}")
    print(
        "Hide AVS submission ratings from planner: "
        f"{args.experiment_filters['hide_avs_submission_ratings_from_planner']}"
    )
    if skipped_tasks:
        print(
            f"Skipping {len(skipped_tasks)} imported tasks with an existing "
            "completed log"
        )
    print(f"Processing {len(tasks)} imported tasks")

    for index, imported_task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {imported_task.name}")
        task_log = await _run_one_task(
            imported_task,
            task_service=task_service,
            loop_factory=default_retrieval_loop,
            run_id=run_id,
            task_index=index,
            tasks_file=tasks_source,
            max_iterations=args.max_iterations,
            max_candidates=max_candidates,
            setting_name=args.setting_name,
            log_dir=log_dir,
            hint_selection=args.experiment_filters["hint_index"],
            hide_avs_submission_ratings_from_planner=args.experiment_filters[
                "hide_avs_submission_ratings_from_planner"
            ],
        )
        task_logs.append(task_log)
        print(
            f"  wrote {task_run_log_path(task_log['log_id'], log_dir)} "
            f"status={task_log['status']}"
        )

    print(
        "Done: "
        f"{sum(1 for log in task_logs if log.get('status') == 'found')} found, "
        f"{sum(1 for log in task_logs if log.get('status') == FOUND_VIA_ANSWERER_STATUS)} "
        "found via answerer, "
        f"{sum(1 for log in task_logs if log.get('status') == 'error')} failed, "
        f"{sum(1 for log in task_logs if log.get('status') == 'timeout')} timed out, "
        f"{len(task_logs)} processed, "
        f"{len(skipped_tasks)} skipped"
    )
    print(f"Batch run id: {run_id}")
    print(f"Log directory: {log_dir}")


async def _run_one_task(
    imported_task: ImportedTaskOption,
    *,
    task_service: Any,
    loop_factory: Any,
    run_id: str,
    task_index: int,
    tasks_file: Path,
    max_iterations: int,
    max_candidates: int,
    setting_name: str,
    log_dir: Path,
    hint_selection: HintSelection = DEFAULT_HINT_INDEX,
    hide_avs_submission_ratings_from_planner: bool = False,
    task_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    start = perf_counter()
    if task_timeout_seconds is None:
        task_timeout_seconds = (
            T_KIS_TASK_TIMEOUT_SECONDS
            if imported_task.task_type == TaskType.TEXTUAL_KIS
            else TASK_TIMEOUT_SECONDS
        )
    payload = imported_task_to_payload(imported_task)
    log_id = f"{run_id}-{task_index:03d}-{imported_task.name}"
    task_log: dict[str, Any] = {
        "log_id": log_id,
        "run_id": run_id,
        "setting_name": setting_name,
        "task_name": imported_task.name,
        "task_type": imported_task.task_type.value,
        "dataset": imported_task.dataset.value,
        "status": "running",
        "created_at": started_at.isoformat(),
        "started_at": started_at.isoformat(),
        "source_file": str(tasks_file),
        "max_iterations": max_iterations,
        "max_candidates": max_candidates,
        "hint_index": hint_selection,
        "hide_avs_submission_ratings_from_planner": (
            hide_avs_submission_ratings_from_planner
        ),
        "timeout_seconds": task_timeout_seconds,
        "iteration_count": 0,
        "final_candidate_id": None,
        "final_answer": None,
        "final_answerer_output": None,
        "final_answerer_validation": None,
        "target_video": (
            imported_task.target_video.model_dump(mode="json")
            if imported_task.target_video
            else None
        ),
        "payload": payload.model_dump(mode="json"),
        "task_input_descriptions": {},
        "actions": [],
    }
    save_task_run_log(task_log, log_dir)
    active_iteration_index: int | None = None
    active_iteration_snapshot: RetrievalLoopIteration | None = None
    completed_iterations: list[RetrievalLoopIteration] = []
    stored_task: Any | None = None

    def record_iteration_start(index: int) -> None:
        nonlocal active_iteration_index, active_iteration_snapshot
        active_iteration_index = index
        active_iteration_snapshot = None

    def record_iteration_progress(iteration: RetrievalLoopIteration) -> None:
        nonlocal active_iteration_snapshot
        active_iteration_snapshot = iteration.model_copy(deep=True)
        _upsert_iteration_log_action(task_log, iteration)
        save_task_run_log(task_log, log_dir)

    def record_iteration_complete(iteration: RetrievalLoopIteration) -> None:
        nonlocal active_iteration_index, active_iteration_snapshot
        snapshot = iteration.model_copy(deep=True)
        completed_iterations.append(snapshot)
        _upsert_iteration_log_action(task_log, snapshot)
        task_log["iteration_count"] = len(task_log["actions"])
        active_iteration_index = None
        active_iteration_snapshot = None
        save_task_run_log(task_log, log_dir)

    try:
        async with asyncio.timeout(task_timeout_seconds):
            stored_task = await task_service.create_task(payload)
            task_log["task_id"] = str(stored_task.id)
            task_log["task_input_descriptions"] = dict(
                getattr(stored_task, "task_input_descriptions", {}) or {}
            )
            save_task_run_log(task_log, log_dir)
            loop = loop_factory(
                max_iterations=max_iterations,
                max_candidates_per_iteration=max_candidates,
                hide_avs_submission_ratings_from_planner=(
                    hide_avs_submission_ratings_from_planner
                ),
            )
            result = await loop.run(
                _payload_from_stored_task(payload),
                encoded_task=stored_task.encoded_task,
                on_iteration_start=record_iteration_start,
                on_iteration_complete=record_iteration_complete,
                on_iteration_progress=record_iteration_progress,
                **(
                    {"task_description_hints": imported_task.input_text_hints}
                    if hint_selection == ITERATIVE_HINT_SELECTION
                    and imported_task.task_type.value == "t-kis"
                    else {}
                ),
            )
        answerer_fallback = _final_answerer_log_fields(result)
        task_log.update(
            {
                "status": (
                    FOUND_VIA_ANSWERER_STATUS
                    if answerer_fallback["accepted"]
                    else result.status.value
                ),
                "iteration_count": len(result.iterations),
                "final_candidate_id": result.final_candidate_id,
                "final_answer": result.final_answer,
                "final_answerer_output": answerer_fallback["output"],
                "final_answerer_validation": answerer_fallback["validation"],
                "final_submission": (
                    result.final_submission.model_dump(mode="json")
                    if result.final_submission
                    else None
                ),
                "actions": [
                    _iteration_log_action(iteration) for iteration in result.iterations
                ],
                "result": result.model_dump(mode="json"),
            }
        )
    except TimeoutError:
        timeout_result = _timeout_loop_result(
            encoded_task=(
                stored_task.encoded_task if stored_task is not None else None
            ),
            completed_iterations=completed_iterations,
            active_iteration_index=active_iteration_index,
            active_iteration=active_iteration_snapshot,
            duration_ms=round((perf_counter() - start) * 1000, 3),
        )
        task_log.update(
            {
                "status": "timeout",
                "iteration_count": len(timeout_result.iterations),
                "actions": [
                    _iteration_log_action(iteration)
                    for iteration in timeout_result.iterations
                ],
                "result": timeout_result.model_dump(mode="json"),
                "error": (
                    f"Task exceeded the {task_timeout_seconds:g}-second runtime limit."
                ),
            }
        )
    except Exception as error:
        task_log.update(
            {
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        task_log["finished_at"] = datetime.now(timezone.utc).isoformat()
        task_log["duration_ms"] = round((perf_counter() - start) * 1000, 3)
        save_task_run_log(task_log, log_dir)

    return task_log


def _iteration_log_action(iteration: RetrievalLoopIteration) -> dict[str, Any]:
    logged_candidate_ids = _iteration_log_candidate_ids(iteration)
    return {
        "iteration": iteration.index,
        "action": iteration.action,
        "endpoint": iteration.endpoint,
        "candidate_count": len(logged_candidate_ids),
        "candidate_ids": logged_candidate_ids,
        "retrieval_error": iteration.retrieval_error,
        "submissions": [
            submission.model_dump(mode="json") for submission in iteration.submissions
        ],
        "answerer_output": (
            iteration.answerer_output.model_dump(mode="json")
            if iteration.answerer_output
            else None
        ),
        "answerer_validation": (
            iteration.answerer_validation.model_dump(mode="json")
            if iteration.answerer_validation
            else None
        ),
        "answerer_error": iteration.answerer_error,
        "vqa_runs": _iteration_vqa_runs(iteration),
        "vqa_planner_sessions": _iteration_vqa_planner_sessions(iteration),
        "timings_ms": iteration.timings_ms,
    }


def _upsert_iteration_log_action(
    task_log: dict[str, Any],
    iteration: RetrievalLoopIteration,
) -> None:
    action = _iteration_log_action(iteration)
    actions = task_log.setdefault("actions", [])
    for index, existing_action in enumerate(actions):
        if (
            isinstance(existing_action, dict)
            and existing_action.get("iteration") == iteration.index
        ):
            actions[index] = action
            break
    else:
        actions.append(action)
    task_log["iteration_count"] = len(actions)


def _iteration_vqa_runs(
    iteration: RetrievalLoopIteration,
) -> list[dict[str, Any]]:
    """Extract complete encoder-selected N/T/V VQA runs for task logs."""

    runs: list[dict[str, Any]] = []
    for evaluation in iteration.evaluator_results:
        raw_attempts = evaluation.metadata.get("vqa_attempts")
        final_result = evaluation.metadata.get("vqa")
        if not isinstance(raw_attempts, list) and not isinstance(final_result, dict):
            continue

        attempt_items = raw_attempts if isinstance(raw_attempts, list) else []
        attempts = [item for item in attempt_items if isinstance(item, dict)]
        final = final_result if isinstance(final_result, dict) else None
        result_for_summary = final or (attempts[-1] if attempts else {})
        result_metadata = result_for_summary.get("metadata")
        if not isinstance(result_metadata, dict):
            result_metadata = {}

        runs.append(
            {
                "candidate_id": evaluation.candidate_id,
                "source_video_id": result_metadata.get("source_video_id"),
                "temporal_mode": result_metadata.get("qa_temporal_mode"),
                "procedure": result_metadata.get("vqa_strategy"),
                "answerable": result_for_summary.get("answerable"),
                "answer": result_for_summary.get("answer"),
                "reason": result_for_summary.get("reason"),
                "answer_validation": evaluation.metadata.get("vqa_answer_validation"),
                "rejected_answers": evaluation.metadata.get("rejected_vqa_answers", []),
                "attempts": attempts,
                "final_result": final,
            }
        )
    return runs


def _iteration_vqa_planner_sessions(
    iteration: RetrievalLoopIteration,
) -> list[dict[str, Any]]:
    """Extract complete VQA-planner operations for incremental task logs."""

    sessions: list[dict[str, Any]] = []
    for evaluation in iteration.evaluator_results:
        session = evaluation.metadata.get("vqa_planner")
        if not isinstance(session, dict):
            continue
        session_metadata = session.get("metadata")
        if not isinstance(session_metadata, dict):
            session_metadata = {}
        raw_steps = session.get("steps")
        steps = raw_steps if isinstance(raw_steps, list) else []
        sessions.append(
            {
                "candidate_id": evaluation.candidate_id,
                "source_video_id": session_metadata.get("source_video_id"),
                "invocation_timestamp_ms": session_metadata.get(
                    "invocation_timestamp_ms"
                ),
                "matched_scene_description": session_metadata.get(
                    "matched_scene_description"
                ),
                "answer": session.get("answer"),
                "ended_without_answer": session.get("ended_without_answer", False),
                "reason": session.get("reason"),
                "steps": [
                    _vqa_planner_step_log(step)
                    for step in steps
                    if isinstance(step, dict)
                ],
                "metadata": session_metadata,
            }
        )
    return sessions


def _vqa_planner_step_log(step: dict[str, Any]) -> dict[str, Any]:
    """Flatten one planner decision while preserving its evidence and traces."""

    decision = step.get("decision")
    if not isinstance(decision, dict):
        decision = {}
    return {
        "index": step.get("index"),
        "action": decision.get("action"),
        "result": step.get("result"),
        "validation": step.get("validation"),
        "planner_metadata": decision.get("metadata", {}),
        "operation_metadata": step.get("metadata", {}),
    }


def _iteration_log_candidate_ids(iteration: RetrievalLoopIteration) -> list[str]:
    """Show the full retrieval window and identify candidates sent to the browser."""

    if iteration.retrieved_candidate_ids is None:
        return iteration.candidate_ids

    browser_candidate_ids = set(iteration.candidate_ids)
    return [
        (
            f"{candidate_id} (sent to browser)"
            if candidate_id in browser_candidate_ids
            else candidate_id
        )
        for candidate_id in iteration.retrieved_candidate_ids
    ]


def _timeout_loop_result(
    *,
    encoded_task: EncodedTask | None,
    completed_iterations: list[RetrievalLoopIteration],
    active_iteration_index: int | None,
    active_iteration: RetrievalLoopIteration | None,
    duration_ms: float,
) -> RetrievalLoopResult:
    iterations = [iteration.model_copy(deep=True) for iteration in completed_iterations]
    if active_iteration is not None:
        iteration = active_iteration.model_copy(deep=True)
        iteration.retrieval_error = iteration.retrieval_error or "Time limit"
        iteration.timings_ms["total"] = duration_ms
        iterations.append(iteration)
    elif active_iteration_index is not None:
        iterations.append(
            RetrievalLoopIteration(
                index=active_iteration_index,
                action="Time limit",
                endpoint="timeout",
                retrieval_error="Time limit",
            )
        )
    return RetrievalLoopResult(
        status=RetrievalLoopStatus.TIMEOUT,
        encoded_task=(
            encoded_task
            if encoded_task is not None
            else EncodedTask(
                status="pending",
                notes=["Task timed out before encoding completed."],
            )
        ),
        iterations=iterations,
        timings_ms={"total": duration_ms},
    )


def _payload_from_stored_task(payload: TaskCreate) -> TaskCreate:
    return TaskCreate(
        task_type=payload.task_type,
        dataset=payload.dataset,
        input=payload.input,
        source_task_name=payload.source_task_name,
    )


def _final_answerer_log_fields(result: RetrievalLoopResult) -> dict[str, Any]:
    """Return the final answerer fallback without changing the retrieval result."""

    if result.status != RetrievalLoopStatus.NOT_FOUND or not result.iterations:
        return {"output": None, "validation": None, "accepted": False}

    final_iteration = result.iterations[-1]
    output = final_iteration.answerer_output
    validation = final_iteration.answerer_validation
    return {
        "output": output.model_dump(mode="json") if output else None,
        "validation": validation.model_dump(mode="json") if validation else None,
        "accepted": (
            validation is not None
            and validation.status == AnswererValidationStatus.ACCEPTED
        ),
    }


def _filter_tasks(
    tasks: list[ImportedTaskOption],
    args: argparse.Namespace,
    experiment_filters: ExperimentSettings,
) -> list[ImportedTaskOption]:
    filtered = tasks
    experiment_task_type = experiment_filters["task_type"]
    if experiment_task_type != "all":
        filtered = [
            task for task in filtered if task.task_type.value == experiment_task_type
        ]
    experiment_dataset = experiment_filters["dataset"]
    if experiment_dataset != "all":
        filtered = [
            task for task in filtered if task.dataset.value == experiment_dataset
        ]
    if args.task_name:
        names = set(args.task_name)
        filtered = [task for task in filtered if task.name in names]
    if args.task_type:
        task_types = set(args.task_type)
        filtered = [task for task in filtered if task.task_type.value in task_types]
    if args.dataset:
        datasets = set(args.dataset)
        filtered = [task for task in filtered if task.dataset.value in datasets]
    if args.limit is not None:
        filtered = filtered[: args.limit]
    return filtered


def _select_tkis_hints(
    tasks: list[ImportedTaskOption],
    hint_index: HintSelection,
) -> list[ImportedTaskOption]:
    """Select one configured T-KIS hint without discarding the other hints."""

    selected_tasks: list[ImportedTaskOption] = []
    for task in tasks:
        if task.task_type.value != "t-kis":
            selected_tasks.append(task)
            continue
        if not task.input_text_hints:
            # A missing array identifies a legacy task file whose ``text`` field
            # already contains the final hint.
            if hint_index == ITERATIVE_HINT_SELECTION:
                raise ValueError(
                    f"T-KIS task {task.name!r} cannot use iterative hints because "
                    "its imported task file has no text_hints array."
                )
            selected_tasks.append(task)
            continue
        selected_hint_index = (
            0 if hint_index == ITERATIVE_HINT_SELECTION else hint_index
        )
        if hint_index == ITERATIVE_HINT_SELECTION and len(task.input_text_hints) != 3:
            raise ValueError(
                f"T-KIS task {task.name!r} must have exactly three text hints "
                "to use iterative hint selection."
            )
        if selected_hint_index >= len(task.input_text_hints):
            raise ValueError(
                f"T-KIS task {task.name!r} has {len(task.input_text_hints)} text "
                f"hints, so hint_index {selected_hint_index} is unavailable."
            )
        selected_tasks.append(
            task.model_copy(
                update={"input_text": task.input_text_hints[selected_hint_index]}
            )
        )
    return selected_tasks


def _skip_completed_tasks(
    tasks: list[ImportedTaskOption],
    log_dir: Path,
) -> tuple[list[ImportedTaskOption], list[ImportedTaskOption]]:
    """Split tasks into runnable and already completed tasks for one setting."""

    completed_task_names = {
        log.task_name
        for log in list_task_run_logs(log_dir)
        if log.status in COMPLETED_TASK_STATUSES
    }
    runnable = [task for task in tasks if task.name not in completed_task_names]
    skipped = [task for task in tasks if task.name in completed_task_names]
    return runnable, skipped


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("imported-tasks-%Y%m%dT%H%M%SZ")


def _task_log_dir(setting_name: str) -> Path:
    return task_run_log_dir() / setting_name


def _load_experiment_filters(settings_dir: Path) -> ExperimentSettings:
    experiments_file = settings_dir / EXPERIMENTS_FILE_NAME
    try:
        with experiments_file.open(encoding="utf-8") as file:
            raw_filters = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(
            f"Missing experiment filter file: {experiments_file}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in experiment filter file {experiments_file}: {error.msg}"
        ) from error

    if not isinstance(raw_filters, dict):
        raise ValueError(f"{experiments_file} must contain a JSON object.")

    task_type = raw_filters.get("task_type")
    dataset = raw_filters.get("dataset")
    max_iterations = raw_filters.get("max_iterations")
    hint_index = raw_filters.get("hint_index", DEFAULT_HINT_INDEX)
    hide_avs_submission_ratings = raw_filters.get(
        "hide_avs_submission_ratings_from_planner",
        False,
    )
    if task_type not in EXPERIMENT_TASK_TYPES:
        raise ValueError(
            f"{experiments_file} task_type must be one of: "
            f"{', '.join(sorted(EXPERIMENT_TASK_TYPES))}."
        )
    if dataset not in EXPERIMENT_DATASETS:
        raise ValueError(
            f"{experiments_file} dataset must be one of: "
            f"{', '.join(sorted(EXPERIMENT_DATASETS))}."
        )
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError(
            f"{experiments_file} max_iterations must be a positive integer."
        )
    invalid_fixed_hint_index = (
        not isinstance(hint_index, int)
        or isinstance(hint_index, bool)
        or hint_index not in EXPERIMENT_HINT_INDICES
    )
    if hint_index != ITERATIVE_HINT_SELECTION and invalid_fixed_hint_index:
        raise ValueError(
            f"{experiments_file} hint_index must be one of: 0, 1, 2, "
            f"or {ITERATIVE_HINT_SELECTION!r}."
        )
    if not isinstance(hide_avs_submission_ratings, bool):
        raise ValueError(
            f"{experiments_file} hide_avs_submission_ratings_from_planner must "
            "be a boolean."
        )
    return {
        "task_type": task_type,
        "dataset": dataset,
        "max_iterations": max_iterations,
        "hint_index": hint_index,
        "hide_avs_submission_ratings_from_planner": hide_avs_submission_ratings,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run imported VBS tasks using a named experiment configuration and "
            "write logs for frontend inspection."
        ),
    )
    parser.add_argument(
        "setting_name",
        help="Name of the experiment settings directory.",
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        help=(
            "Path to one downloaded imported tasks JSON file. By default, all "
            "vbsYEAR_tasks.json files in the scripts directory are processed."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Optional stable id for the created log file.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help=(
            "Maximum retrieval loop iterations per task. Overrides max_iterations "
            "from the selected experiments.json."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        help=(
            "Maximum candidates loaded per loop iteration. Defaults to the "
            "number_of_candidates in the selected experiment settings."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N imported tasks after filtering.",
    )
    parser.add_argument(
        "--task-name",
        action="append",
        help="Only process a specific imported task name. Can be repeated.",
    )
    parser.add_argument(
        "--task-type",
        action="append",
        choices=["v-kis", "t-kis", "avs", "qa"],
        help="Only process one task type. Can be repeated.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["mvk", "v3c", "lhe"],
        help="Only process one dataset. Can be repeated.",
    )
    args = parser.parse_args(argv)
    try:
        settings_dir = experiment_settings_dir(args.setting_name)
        args.experiment_filters = _load_experiment_filters(settings_dir)
        if args.max_iterations is None:
            args.max_iterations = args.experiment_filters["max_iterations"]
        elif args.max_iterations <= 0:
            raise ValueError("--max-iterations must be a positive integer.")
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    main()
