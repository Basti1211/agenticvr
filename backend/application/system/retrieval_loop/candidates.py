"""Candidate conversion and image resolution helpers for retrieval loops."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from tempfile import gettempdir
from urllib.parse import urljoin, urlparse

from application.modalities.image import download_image
from application.modalities.video import extract_video_frame, resize_video
from application.settings.settings import (
    _browser_settings,
    _data_layer_settings,
    _qa_settings,
)
from application.system.output_lock import output_lock
from application.system.timing import elapsed_ms, timer_start
from application.system.vlm.candidate_video_aliases import (
    register_candidate_video_id,
)
from application.tasks.data_layer_paths import dataset_media_path, normalize_media_uri_path
from application.tasks.models import Dataset, MediaInput, TaskCreate, VideoSegment
from application.types.browser.models import BrowserCandidate
from application.types.retrieval.candidate import Candidate, CandidateFrame
from application.types.vlm.prompt import ImageInput, VideoInput


CandidateImageResolver = Callable[[Candidate], ImageInput]
CandidateVideoResolver = Callable[[Candidate, Dataset], VideoInput]
DirectAnswerResolver = Callable[[str, int, Dataset], Candidate]
QAVideoResolver = Callable[[str, int, int, Dataset], VideoInput]
TaskReferenceVideoResolver = Callable[[TaskCreate], VideoInput | None]
SourceVideoDurationResolver = Callable[[Candidate, Dataset], float]

BROWSER_CLIP_DIR = Path(gettempdir()) / "vlm-prak" / "browser-clips"
DIRECT_ANSWER_FRAME_DIR = Path(gettempdir()) / "vlm-prak" / "direct-answer-frames"
QA_CLIP_DIR = Path(gettempdir()) / "vlm-prak" / "qa-clips"
SECOND_STAGE_EVALUATOR_CLIP_DIR = (
    Path(gettempdir()) / "vlm-prak" / "second-stage-evaluator-clips"
)

logger = logging.getLogger(__name__)


def candidate_to_browser_candidate(
    candidate: Candidate,
    dataset: Dataset,
    video_resolver: CandidateVideoResolver | None = None,
    image_resolver: CandidateImageResolver | None = None,
) -> BrowserCandidate:
    """Convert a retrieved keyframe into temporal evidence for the browser.

    Retrieval remains keyframe-based. The browser instead receives the retrieved
    scene as a video clip, so it can inspect local motion and temporal order.
    ``image_resolver`` remains as a compatibility override for callers that
    deliberately provide image-only browser evidence (notably test doubles);
    normal production calls use video evidence.
    """

    if candidate.frames:
        return BrowserCandidate(
            images=candidate_image_inputs(candidate),
            metadata=candidate.model_dump(mode="json"),
        )

    if video_resolver is None and image_resolver is not None:
        return BrowserCandidate(
            image=image_resolver(candidate),
            metadata=candidate.model_dump(mode="json"),
        )

    return BrowserCandidate(
        video=(video_resolver or candidate_video_input)(candidate, dataset),
        metadata=candidate.model_dump(mode="json"),
    )


def candidate_to_browser_image_candidate(
    candidate: Candidate,
    image_resolver: CandidateImageResolver | None = None,
) -> BrowserCandidate:
    """Convert a retrieved keyframe into image evidence for browser screening."""

    if candidate.frames:
        return BrowserCandidate(
            images=candidate_image_inputs(candidate),
            metadata=candidate.model_dump(mode="json"),
        )

    return BrowserCandidate(
        image=(image_resolver or candidate_image_input)(candidate),
        metadata=candidate.model_dump(mode="json"),
    )


def candidate_to_browser_video_candidate(
    candidate: Candidate,
    dataset: Dataset,
    video_resolver: CandidateVideoResolver | None = None,
    *,
    scene_interval_ms: tuple[int, int] | None = None,
) -> BrowserCandidate:
    """Convert a screened candidate into temporal evidence for browser inspection."""

    if video_resolver is not None:
        video = video_resolver(candidate, dataset)
    else:
        video = candidate_video_input(
            candidate,
            dataset,
            scene_interval_ms=scene_interval_ms,
        )
    return BrowserCandidate(
        video=video,
        metadata=candidate.model_dump(mode="json"),
    )


def candidate_image_input(candidate: Candidate) -> ImageInput:
    """Resolve a candidate URI into the local image input expected by VLM modules."""

    return _image_input(
        identifier=candidate_id(candidate),
        uri=candidate.uri,
        log_candidate_id=candidate_id(candidate),
    )


def candidate_image_inputs(candidate: Candidate) -> list[ImageInput]:
    """Resolve all ordered keyframes represented by a candidate."""

    if not candidate.frames:
        return [candidate_image_input(candidate)]

    return [
        _frame_image_input(frame, group_identifier=candidate_id(candidate))
        for frame in candidate.frames
    ]


def _frame_image_input(
    frame: CandidateFrame,
    *,
    group_identifier: str,
) -> ImageInput:
    frame_identifier = "_".join(frame.id)
    return _image_input(
        identifier=frame_identifier,
        uri=frame.uri,
        log_candidate_id=group_identifier,
    )


def _image_input(
    *,
    identifier: str,
    uri: str,
    log_candidate_id: str,
) -> ImageInput:
    """Resolve an image URI into the local image input expected by VLM modules."""

    total_start = timer_start()
    parsed_uri = urlparse(uri)

    if parsed_uri.scheme in {"http", "https"}:
        source_kind = "remote_uri"
        image_path = download_image(uri)
    elif parsed_uri.scheme == "file":
        source_kind = "file_uri"
        image_path = Path(parsed_uri.path)
    else:
        local_path = Path(uri)
        if local_path.exists():
            source_kind = "local_path"
            image_path = local_path
        else:
            source_kind = "data_layer_url"
            image_path = download_image(_image_url(uri))

    logger.info(
        "timing.browser.image_resolve.done candidate_id=%s source_kind=%s "
        "file_bytes=%s total_ms=%.3f",
        log_candidate_id,
        source_kind,
        _file_size(image_path),
        elapsed_ms(total_start),
    )
    return ImageInput(identifier=identifier, file=image_path)


def candidate_video_input(
    candidate: Candidate,
    dataset: Dataset,
    *,
    scene_interval_ms: tuple[int, int] | None = None,
) -> VideoInput:
    """Materialize one bounded scene window for authoritative browser inspection."""

    start_ms, end_ms = scene_interval_ms or candidate_review_scene_interval_ms(
        candidate,
        dataset,
    )

    source = _video_url(candidate, dataset)
    frame_rate, resolution_width, resolution_height = _browser_clip_settings()
    timeout_seconds, retry_count = _browser_clip_process_settings()
    start_seconds = start_ms / 1000
    duration_seconds = (end_ms - start_ms) / 1000
    clip_path = _browser_clip_path(
        source=source,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        frame_rate=frame_rate,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
    )

    total_start = timer_start()
    with output_lock(clip_path):
        cache_hit = clip_path.is_file() and clip_path.stat().st_size > 0
        logger.info(
            "timing.browser.video_resolve.start candidate_id=%s cache_hit=%s "
            "start_seconds=%.3f duration_seconds=%.3f fps=%s "
            "resolution=%sx%s",
            candidate_id(candidate),
            cache_hit,
            start_seconds,
            duration_seconds,
            frame_rate,
            resolution_width,
            resolution_height,
        )
        if not cache_hit:
            resize_video(
                source,
                clip_path,
                resolution_width,
                resolution_height,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                frame_rate=frame_rate,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
            )

    logger.info(
        "timing.browser.video_resolve.done candidate_id=%s cache_hit=%s "
        "file_bytes=%s total_ms=%.3f",
        candidate_id(candidate),
        cache_hit,
        _file_size(clip_path),
        elapsed_ms(total_start),
    )
    return VideoInput(identifier=candidate_id(candidate), file=clip_path)


def candidate_centered_video_input(
    candidate: Candidate,
    dataset: Dataset,
    *,
    before_seconds: float = 5,
    after_seconds: float = 5,
    frame_rate: int = 5,
) -> tuple[VideoInput, int, int]:
    """Materialize a fixed window around one candidate keyframe.

    The requested window is truncated, rather than shifted, at the source-video
    boundaries so it never includes more than ``before_seconds`` before or
    ``after_seconds`` after the keyframe.
    """

    if before_seconds < 0 or after_seconds < 0:
        raise ValueError("Candidate context durations must not be negative.")
    if before_seconds + after_seconds <= 0:
        raise ValueError("Candidate context duration must be greater than 0.")
    if frame_rate <= 0:
        raise ValueError("Candidate context frame rate must be greater than 0.")
    if candidate.time_keyframe < 0:
        raise ValueError("Candidate keyframe timestamp must not be negative.")

    source_duration_ms = round(
        candidate_source_video_duration_seconds(candidate, dataset) * 1000
    )
    if candidate.time_keyframe > source_duration_ms:
        raise ValueError("Candidate keyframe exceeds the source video duration.")

    start_ms = max(0, candidate.time_keyframe - round(before_seconds * 1000))
    end_ms = min(
        source_duration_ms,
        candidate.time_keyframe + round(after_seconds * 1000),
    )
    if end_ms <= start_ms:
        raise ValueError("Candidate context window has no positive duration.")

    source = _video_url(candidate, dataset)
    _, resolution_width, resolution_height = _browser_clip_settings()
    timeout_seconds, retry_count = _browser_clip_process_settings()
    clip_path = _second_stage_evaluator_clip_path(
        source=source,
        start_ms=start_ms,
        end_ms=end_ms,
        frame_rate=frame_rate,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
    )

    total_start = timer_start()
    with output_lock(clip_path):
        cache_hit = clip_path.is_file() and clip_path.stat().st_size > 0
        if not cache_hit:
            resize_video(
                source,
                clip_path,
                resolution_width,
                resolution_height,
                start_seconds=start_ms / 1000,
                duration_seconds=(end_ms - start_ms) / 1000,
                frame_rate=frame_rate,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
            )

    logger.info(
        "timing.second_stage_evaluator.video_resolve.done candidate_id=%s "
        "cache_hit=%s window_start_ms=%s window_end_ms=%s fps=%s "
        "file_bytes=%s total_ms=%.3f",
        candidate_id(candidate),
        cache_hit,
        start_ms,
        end_ms,
        frame_rate,
        _file_size(clip_path),
        elapsed_ms(total_start),
    )
    return (
        VideoInput(identifier=candidate_id(candidate), file=clip_path),
        start_ms,
        end_ms,
    )


def direct_answer_candidate(
    video_id: str,
    timestamp_ms: int,
    dataset: Dataset,
) -> Candidate:
    """Materialize a planner-submitted video frame as a retrieval candidate."""

    if timestamp_ms < 0:
        raise ValueError("Direct Answer timestamp must not be negative.")

    source = _video_url_for_id(video_id, dataset)
    frame_path = _direct_answer_frame_path(source, timestamp_ms)
    with output_lock(frame_path):
        if not frame_path.is_file() or frame_path.stat().st_size == 0:
            extract_video_frame(source, frame_path, timestamp_ms / 1000)

    return Candidate(
        uri=frame_path.resolve().as_uri(),
        rank=None,
        score=None,
        id=[video_id, str(timestamp_ms)],
        time_keyframe=timestamp_ms,
        time_scene_start=timestamp_ms,
        time_scene_end=timestamp_ms,
    )


def candidate_source_video_duration_seconds(
    candidate: Candidate,
    dataset: Dataset,
) -> float:
    """Read the duration of the candidate's complete parent video."""

    source = _video_url(candidate, dataset)
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                source,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "ffprobe is required to inspect source-video duration."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Timed out probing source video {candidate.id[0]!r}."
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            f"Could not probe source video {candidate.id[0]!r}: "
            f"{detail or 'ffprobe failed'}"
        ) from error

    try:
        payload = json.loads(completed.stdout)
        duration_seconds = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid duration metadata for source video {candidate.id[0]!r}."
        ) from error

    if duration_seconds <= 0:
        raise RuntimeError(
            f"Source video {candidate.id[0]!r} has invalid duration "
            f"{duration_seconds}."
        )

    return duration_seconds


def qa_video_input(
    video_name: str,
    start_ms: int,
    end_ms: int,
    dataset: Dataset,
) -> VideoInput:
    """Materialize and cache the exact video interval used by planner QA."""

    return _qa_or_vqa_video_input(
        video_name,
        start_ms,
        end_ms,
        dataset,
        settings_key_prefix="",
    )


def vqa_video_input(
    video_name: str,
    start_ms: int,
    end_ms: int,
    dataset: Dataset,
) -> VideoInput:
    """Materialize and cache a high-quality exact interval for final VQA."""

    return _qa_or_vqa_video_input(
        video_name,
        start_ms,
        end_ms,
        dataset,
        settings_key_prefix="vqa_",
    )


def _qa_or_vqa_video_input(
    video_name: str,
    start_ms: int,
    end_ms: int,
    dataset: Dataset,
    *,
    settings_key_prefix: str,
) -> VideoInput:
    """Materialize and cache an exact QA/VQA video interval."""

    if start_ms < 0:
        raise ValueError("VQA start_ms must not be negative.")
    if end_ms <= start_ms:
        raise ValueError("VQA end_ms must be greater than start_ms.")

    source = _video_url_for_id(video_name, dataset)
    fps, resolution_width, resolution_height, crf = _qa_clip_settings(
        settings_key_prefix
    )
    clip_path = _qa_clip_path(
        source=source,
        start_ms=start_ms,
        end_ms=end_ms,
        frame_rate=fps,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
        crf=crf,
        profile=settings_key_prefix.rstrip("_") or "qa",
    )
    with output_lock(clip_path):
        if not clip_path.is_file() or clip_path.stat().st_size == 0:
            resize_video(
                source,
                clip_path,
                resolution_width,
                resolution_height,
                start_seconds=start_ms / 1000,
                duration_seconds=(end_ms - start_ms) / 1000,
                frame_rate=fps,
                crf=crf,
            )

    return VideoInput(
        identifier=f"{video_name}:{start_ms}-{end_ms}",
        file=clip_path,
    )


def task_reference_video_input(task: TaskCreate) -> VideoInput | None:
    """Resolve the task/reference video that may be shown to the evaluator."""

    if task.input.task_video is not None:
        return _task_video_input(task.input.task_video, task.dataset)

    media = task.input.media
    if media is not None and media.media_type == "video":
        return _uploaded_video_input(media)

    return None


def _browser_clip_settings() -> tuple[int, int, int]:
    settings = _browser_settings()
    frame_rate = int(settings["clip_frame_rate"])
    resolution_width = int(settings["clip_resolution_width"])
    resolution_height = int(settings["clip_resolution_height"])
    if frame_rate <= 0:
        raise ValueError("Browser clip frame rate must be greater than 0.")
    if resolution_width <= 0 or resolution_height <= 0:
        raise ValueError("Browser clip resolution must be greater than 0.")
    return frame_rate, resolution_width, resolution_height


def _browser_clip_process_settings() -> tuple[float, int]:
    settings = _browser_settings()
    timeout_seconds = float(settings.get("clip_timeout_seconds", 10))
    retry_count = int(settings.get("clip_retry_count", 3))
    if timeout_seconds <= 0:
        raise ValueError("Browser clip timeout must be greater than 0.")
    if retry_count < 0:
        raise ValueError("Browser clip retry count must not be negative.")
    return timeout_seconds, retry_count


def _qa_clip_settings(settings_key_prefix: str = "") -> tuple[int, int, int, int]:
    settings = _qa_settings()
    fps = int(settings[f"{settings_key_prefix}fps"])
    resolution_width = int(settings[f"{settings_key_prefix}resolution_width"])
    resolution_height = int(settings[f"{settings_key_prefix}resolution_height"])
    crf = int(settings.get(f"{settings_key_prefix}crf", 23))
    if fps <= 0:
        raise ValueError("QA video fps must be greater than 0.")
    if resolution_width <= 0 or resolution_height <= 0:
        raise ValueError("QA video resolution must be greater than 0.")
    if not 0 <= crf <= 51:
        raise ValueError("QA video CRF must be between 0 and 51.")
    return fps, resolution_width, resolution_height, crf


def _minimum_review_duration_ms() -> int:
    return 5_000


def browser_scene_window_settings_ms() -> tuple[int, int, int]:
    """Return browser scene padding, minimum duration, and maximum duration."""

    settings = _browser_settings()
    values_seconds = (
        float(settings.get("scene_padding_seconds", 2)),
        float(settings.get("minimum_scene_duration_seconds", 12)),
        float(settings.get("maximum_scene_duration_seconds", 24)),
    )
    values_ms = tuple(value * 1000 for value in values_seconds)
    if any(value < 0 for value in values_ms[:1]):
        raise ValueError("Browser scene padding must not be negative.")
    if values_ms[1] <= 0 or values_ms[2] <= 0:
        raise ValueError("Browser scene durations must be greater than zero.")
    if values_ms[1] > values_ms[2]:
        raise ValueError(
            "Browser minimum scene duration must not exceed its maximum duration."
        )
    if any(not value.is_integer() for value in values_ms):
        raise ValueError("Browser scene window settings must resolve to whole milliseconds.")
    return tuple(int(value) for value in values_ms)  # type: ignore[return-value]


def candidate_browser_scene_interval_ms(
    candidate: Candidate,
    dataset: Dataset,
    *,
    source_duration_ms: int | None = None,
) -> tuple[int, int]:
    """Build a padded, bounded, keyframe-centred browser scene window."""

    scene_start = candidate.time_scene_start
    scene_end = candidate.time_scene_end
    keyframe = (
        round(
            (
                min(frame.time_keyframe for frame in candidate.frames)
                + max(frame.time_keyframe for frame in candidate.frames)
            )
            / 2
        )
        if candidate.frames
        else candidate.time_keyframe
    )
    if scene_start < 0 or keyframe < 0:
        raise ValueError("Candidate scene timestamps must not be negative.")
    if scene_end <= scene_start:
        raise ValueError("Candidate scene end must be greater than scene start.")

    if source_duration_ms is None:
        source_duration_ms = round(
            candidate_source_video_duration_seconds(candidate, dataset) * 1000
        )
    if source_duration_ms <= 0:
        raise ValueError("Candidate source video duration must be greater than zero.")
    if scene_end > source_duration_ms:
        raise ValueError("Candidate scene end exceeds the source video duration.")

    padding_ms, minimum_ms, maximum_ms = browser_scene_window_settings_ms()
    padded_start = max(0, scene_start - padding_ms)
    padded_end = min(source_duration_ms, scene_end + padding_ms)

    if padded_end - padded_start > maximum_ms:
        start_ms = keyframe - maximum_ms // 2
        end_ms = start_ms + maximum_ms
    else:
        start_ms, end_ms = padded_start, padded_end

    desired_minimum_ms = min(minimum_ms, source_duration_ms)
    if end_ms - start_ms < desired_minimum_ms:
        missing_ms = desired_minimum_ms - (end_ms - start_ms)
        start_ms -= missing_ms // 2
        end_ms += missing_ms - missing_ms // 2

    if start_ms < 0:
        end_ms = min(source_duration_ms, end_ms - start_ms)
        start_ms = 0
    if end_ms > source_duration_ms:
        start_ms = max(0, start_ms - (end_ms - source_duration_ms))
        end_ms = source_duration_ms

    if end_ms - start_ms > maximum_ms:
        end_ms = min(source_duration_ms, start_ms + maximum_ms)
        start_ms = max(0, end_ms - maximum_ms)

    logger.info(
        "timing.candidate.browser_scene_window candidate_id=%s "
        "original_start_ms=%s original_end_ms=%s keyframe_ms=%s "
        "window_start_ms=%s window_end_ms=%s source_duration_ms=%s",
        candidate_id(candidate),
        scene_start,
        scene_end,
        keyframe,
        start_ms,
        end_ms,
        source_duration_ms,
    )
    return start_ms, end_ms


def candidate_review_scene_interval_ms(
    candidate: Candidate,
    dataset: Dataset,
) -> tuple[int, int]:
    """Return the shared browser/evaluator scene window for a candidate.

    Short scenes extend forward to the configured minimum duration. Near the
    end of a source video the window shifts backward so its duration is retained.
    """

    start_ms = candidate.time_scene_start
    end_ms = candidate.time_scene_end
    if start_ms < 0:
        raise ValueError("Candidate scene start must not be negative.")
    if end_ms <= start_ms:
        raise ValueError("Candidate scene end must be greater than scene start.")

    minimum_duration_ms = _minimum_review_duration_ms()
    scene_duration_ms = end_ms - start_ms
    if scene_duration_ms >= minimum_duration_ms:
        return start_ms, end_ms

    source_duration_ms = round(
        candidate_source_video_duration_seconds(candidate, dataset) * 1000
    )
    if source_duration_ms < end_ms:
        raise ValueError(
            "Candidate scene end exceeds the source video duration."
        )
    desired_duration_ms = min(minimum_duration_ms, source_duration_ms)
    expanded_end_ms = min(source_duration_ms, start_ms + desired_duration_ms)
    expanded_start_ms = max(0, expanded_end_ms - desired_duration_ms)

    logger.info(
        "timing.candidate.scene_expand candidate_id=%s "
        "original_start_ms=%s original_end_ms=%s expanded_start_ms=%s "
        "expanded_end_ms=%s minimum_duration_ms=%s source_duration_ms=%s",
        candidate_id(candidate),
        start_ms,
        end_ms,
        expanded_start_ms,
        expanded_end_ms,
        minimum_duration_ms,
        source_duration_ms,
    )
    return expanded_start_ms, expanded_end_ms


def candidate_id(candidate: Candidate) -> str:
    if candidate.id:
        register_candidate_video_id(candidate.id[0])
    return "_".join(candidate.id)


def candidate_ids(candidates: list[Candidate]) -> list[str]:
    return [candidate_id(candidate) for candidate in candidates]


def _image_url(uri: str) -> str:
    data_layer_settings = _data_layer_settings()
    return urljoin(data_layer_settings["image_path"], normalize_media_uri_path(uri))


def _video_url(candidate: Candidate, dataset: Dataset) -> str:
    if not candidate.id:
        raise ValueError("Candidate requires a video identifier to build a clip.")

    return _video_url_for_id(candidate.id[0], dataset)


def _video_url_for_id(video_id: str, dataset: Dataset) -> str:
    video_name = video_id.strip().strip("/")
    if not video_name:
        raise ValueError("Candidate video identifier must not be empty.")
    if not video_name.lower().endswith(".mp4"):
        video_name = f"{video_name}.mp4"

    data_layer_settings = _data_layer_settings()
    return urljoin(
        data_layer_settings["video_path"].rstrip("/") + "/",
        f"{dataset_media_path(dataset)}/{video_name}",
    )


def _task_video_input(segment: VideoSegment, dataset: Dataset) -> VideoInput:
    start_ms = segment.start_ms or 0
    if segment.end_ms is not None:
        return qa_video_input(segment.name, start_ms, segment.end_ms, dataset)

    source = _video_url_for_id(segment.name, dataset)
    return VideoInput(
        identifier=f"{Path(segment.name).stem}:{start_ms}-end",
        file=Path(source),
    )


def _uploaded_video_input(media: MediaInput) -> VideoInput | None:
    if not media.local_path:
        return None
    return VideoInput(
        identifier=Path(media.name).stem,
        file=Path(media.local_path),
    )


def _direct_answer_frame_path(source: str, timestamp_ms: int) -> Path:
    digest = sha256(f"{source}:{timestamp_ms}".encode("utf-8")).hexdigest()[:16]
    return DIRECT_ANSWER_FRAME_DIR / f"direct-answer-{digest}.jpg"


def _browser_clip_path(
    *,
    source: str,
    start_seconds: float,
    duration_seconds: float,
    frame_rate: int,
    resolution_width: int,
    resolution_height: int,
) -> Path:
    digest = sha256(
        (
            f"{source}:{start_seconds:.6f}:{duration_seconds:.6f}:"
            f"{resolution_width}x{resolution_height}:{frame_rate}fps"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return BROWSER_CLIP_DIR / f"browser-clip-{digest}.mp4"


def _qa_clip_path(
    *,
    source: str,
    start_ms: int,
    end_ms: int,
    frame_rate: int,
    resolution_width: int,
    resolution_height: int,
    crf: int,
    profile: str,
) -> Path:
    digest = sha256(
        (
            f"{profile}:{source}:{start_ms}:{end_ms}:"
            f"{resolution_width}x{resolution_height}:{frame_rate}fps:crf{crf}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return QA_CLIP_DIR / f"qa-clip-{digest}.mp4"


def _second_stage_evaluator_clip_path(
    *,
    source: str,
    start_ms: int,
    end_ms: int,
    frame_rate: int,
    resolution_width: int,
    resolution_height: int,
) -> Path:
    digest = sha256(
        (
            f"{source}:{start_ms}:{end_ms}:"
            f"{resolution_width}x{resolution_height}:{frame_rate}fps"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return SECOND_STAGE_EVALUATOR_CLIP_DIR / f"second-stage-clip-{digest}.mp4"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
