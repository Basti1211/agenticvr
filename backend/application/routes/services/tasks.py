"""Task submission and retrieval endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlparse
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import ValidationError

from application.tasks.media_storage import (
    infer_media_type,
    public_task_media_name,
    save_task_media_upload,
)
from application.tasks.models import (
    Dataset,
    ImportedTaskOption,
    MediaInput,
    TaskCreate,
    TaskOptionsResponse,
    TaskRepresentation,
    TaskRunLog,
    TaskRunLogSummary,
)
from application.tasks.imported import load_downloaded_tasks
from application.tasks.logs import get_task_run_log, list_task_run_logs
from application.tasks.options import get_task_options
from application.tasks.service import task_service
from application.system.retrieval_loop.candidates import (
    direct_answer_candidate,
    qa_video_input,
)
from application.system.retrieval_loop import (
    DEFAULT_MAX_CANDIDATES_PER_ITERATION,
    default_retrieval_loop,
)
from application.types.retrieval_loop import RetrievalLoopResult


router = APIRouter()


def _encoded_frame_metadata(
    task: TaskRepresentation,
    frame_id: str,
) -> dict | None:
    frames = task.encoded_task.metadata.get("selected_video_frames", [])
    if not isinstance(frames, list):
        return None
    for frame in frames:
        if isinstance(frame, dict) and frame.get("id") == frame_id:
            return frame
    return None


@router.get(
    "/options",
    response_model=TaskOptionsResponse,
    summary="Get supported task types and datasets",
)
async def get_options() -> TaskOptionsResponse:
    """Return task setup metadata for frontend and script clients."""

    return get_task_options()


@router.get(
    "/imported",
    response_model=list[ImportedTaskOption],
    summary="Get normalized downloaded task options",
)
async def get_imported_tasks() -> list[ImportedTaskOption]:
    """Return supported downloaded VBS tasks normalized for the frontend."""

    return load_downloaded_tasks()


@router.post(
    "",
    response_model=TaskRepresentation,
    status_code=status.HTTP_201_CREATED,
    summary="Create a task representation",
)
async def create_task(payload: TaskCreate) -> TaskRepresentation:
    """Store a task representation and pass it through the encoder boundary."""

    return await task_service.create_task(payload)


@router.post(
    "/upload",
    response_model=TaskRepresentation,
    status_code=status.HTTP_201_CREATED,
    summary="Create a task representation with uploaded media",
)
async def create_task_with_upload(
    payload: Annotated[str, Form(description="JSON-encoded TaskCreate payload.")],
    media: Annotated[UploadFile, File(description="Image or video file to encode.")],
) -> TaskRepresentation:
    """Store uploaded task media and pass the task through the encoder boundary."""

    try:
        task_payload = TaskCreate.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc

    media_type = infer_media_type(media.content_type)
    if media_type is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded task media must be an image or video.",
        )

    media_path, size_bytes = await save_task_media_upload(media)
    task_media = MediaInput(
        name=public_task_media_name(media_type, media_path),
        media_type=media_type,
        mime_type=media.content_type,
        size_bytes=size_bytes,
        local_path=str(media_path),
    )
    task_payload = task_payload.model_copy(
        update={
            "input": task_payload.input.model_copy(update={"media": task_media}),
        },
    )

    return await task_service.create_task(task_payload)


@router.post(
    "/run",
    response_model=RetrievalLoopResult,
    summary="Create and run a task through the automated retrieval loop",
)
async def run_task_payload(
    payload: TaskCreate,
    max_iterations: int = Query(default=1, ge=1, le=20),
    max_candidates: int = Query(
        default=DEFAULT_MAX_CANDIDATES_PER_ITERATION,
        ge=1,
        le=DEFAULT_MAX_CANDIDATES_PER_ITERATION,
    ),
) -> RetrievalLoopResult:
    """Encode a task and execute the planner/retrieval/browser/verifier loop."""

    loop = default_retrieval_loop(
        max_iterations=max_iterations,
        max_candidates_per_iteration=max_candidates,
    )
    return await loop.run(payload)


@router.get(
    "",
    response_model=list[TaskRepresentation],
    summary="List stored task representations",
)
async def list_tasks() -> list[TaskRepresentation]:
    """Return all task representations stored in this backend process."""

    return task_service.list_tasks()


@router.get(
    "/logs",
    response_model=list[TaskRunLogSummary],
    summary="List imported task batch run logs",
)
async def list_imported_task_run_logs() -> list[TaskRunLogSummary]:
    """Return summaries of imported task batch logs available to the backend."""

    return list_task_run_logs()


@router.get(
    "/logs/{log_id}",
    response_model=TaskRunLog,
    summary="Get an imported task execution log",
)
async def get_imported_task_run_log(log_id: str) -> TaskRunLog:
    """Return the full imported task execution log for one log id."""

    log = get_task_run_log(log_id)
    if log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task run log not found",
        )
    return log


@router.get(
    "/submission-frame",
    response_class=FileResponse,
    summary="Get a submitted video/timestamp frame preview",
)
async def get_submission_frame(
    dataset: Dataset = Query(description="Dataset containing the submitted video."),
    video_id: str = Query(min_length=1, description="Submitted source video id."),
    timestamp_ms: int = Query(ge=0, description="Submitted timestamp in milliseconds."),
) -> FileResponse:
    """Materialize the frame represented by a submitted video/time answer."""

    try:
        frame_path = await run_in_threadpool(
            _submission_frame_path,
            dataset,
            video_id,
            timestamp_ms,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not render submitted frame: {exc}",
        ) from exc

    return FileResponse(frame_path, media_type="image/jpeg")


def _submission_frame_path(
    dataset: Dataset,
    video_id: str,
    timestamp_ms: int,
) -> Path:
    candidate = direct_answer_candidate(video_id, timestamp_ms, dataset)
    parsed_uri = urlparse(candidate.uri)
    frame_source = parsed_uri.path if parsed_uri.scheme == "file" else candidate.uri
    frame_path = Path(unquote(frame_source))
    if not frame_path.is_file():
        raise FileNotFoundError("Submitted frame image was not created.")

    return frame_path


@router.get(
    "/answer-scene",
    response_class=FileResponse,
    summary="Get the known correct answer scene clip",
)
async def get_answer_scene(
    dataset: Dataset = Query(description="Dataset containing the answer video."),
    video_id: str = Query(min_length=1, description="Known answer video id."),
    start_ms: int = Query(ge=0, description="Answer scene start in milliseconds."),
    end_ms: int = Query(ge=0, description="Answer scene end in milliseconds."),
) -> FileResponse:
    """Materialize the known target/answer scene as a browser-previewable clip."""

    try:
        scene_path = await run_in_threadpool(
            _answer_scene_path,
            dataset,
            video_id,
            start_ms,
            end_ms,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not render answer scene: {exc}",
        ) from exc

    return FileResponse(scene_path, media_type="video/mp4")


def _answer_scene_path(
    dataset: Dataset,
    video_id: str,
    start_ms: int,
    end_ms: int,
) -> Path:
    video = qa_video_input(video_id, start_ms, end_ms, dataset)
    if not video.file.is_file():
        raise FileNotFoundError("Answer scene video was not created.")
    return video.file


@router.get(
    "/{task_id}/frames/{frame_id}",
    response_class=FileResponse,
    summary="Get a sampled task encoder video frame",
)
async def get_encoded_task_frame(task_id: UUID, frame_id: str) -> FileResponse:
    """Return one frame selected by the task encoder for frontend preview."""

    task = task_service.get_task(task_id)
    if task is None or task.encoded_task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task frame not found",
        )

    frame = _encoded_frame_metadata(task, frame_id)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task frame not found",
        )

    frame_path = frame.get("file")
    if not isinstance(frame_path, str):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task frame not found",
        )

    path = Path(frame_path)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task frame file not found",
        )

    return FileResponse(path, media_type="image/jpeg")


@router.get(
    "/{task_id}",
    response_model=TaskRepresentation,
    summary="Get a stored task representation",
)
async def get_task(task_id: UUID) -> TaskRepresentation:
    """Return one stored task representation by id."""

    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )
    return task


@router.post(
    "/{task_id}/run",
    response_model=RetrievalLoopResult,
    summary="Run a stored task through the automated retrieval loop",
)
async def run_stored_task(
    task_id: UUID,
    max_iterations: int = Query(default=1, ge=1, le=20),
    max_candidates: int = Query(
        default=DEFAULT_MAX_CANDIDATES_PER_ITERATION,
        ge=1,
        le=DEFAULT_MAX_CANDIDATES_PER_ITERATION,
    ),
) -> RetrievalLoopResult:
    """Execute the retrieval loop for a task already stored by the task service."""

    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    payload = TaskCreate(
        task_type=task.task_type,
        dataset=task.dataset,
        input=task.input,
        source_task_name=task.source_task_name,
    )
    loop = default_retrieval_loop(
        max_iterations=max_iterations,
        max_candidates_per_iteration=max_candidates,
    )
    result = await loop.run(payload, encoded_task=task.encoded_task)
    return task_service.set_loop_result(task_id, result)


@router.post(
    "/{task_id}/continue",
    response_model=RetrievalLoopResult,
    summary="Continue a stored task retrieval loop from previous history",
)
async def continue_stored_task(
    task_id: UUID,
    max_iterations: int = Query(default=1, ge=1, le=20),
    max_candidates: int = Query(
        default=DEFAULT_MAX_CANDIDATES_PER_ITERATION,
        ge=1,
        le=DEFAULT_MAX_CANDIDATES_PER_ITERATION,
    ),
) -> RetrievalLoopResult:
    """Execute more loop iterations using the previous planner/history context."""

    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    previous_result = task_service.get_loop_result(task_id)
    if previous_result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Task has no retrieval loop history. Run the task before continuing.",
        )

    payload = TaskCreate(
        task_type=task.task_type,
        dataset=task.dataset,
        input=task.input,
        source_task_name=task.source_task_name,
    )
    loop = default_retrieval_loop(
        max_iterations=max_iterations,
        max_candidates_per_iteration=max_candidates,
    )
    result = await loop.run(
        payload,
        encoded_task=task.encoded_task,
        previous_result=previous_result,
    )
    return task_service.set_loop_result(task_id, result)
