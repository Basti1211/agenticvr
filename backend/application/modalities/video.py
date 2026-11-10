from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import gettempdir
from urllib.parse import urlparse

from application.system.timing import elapsed_ms, timer_start


DEFAULT_VIDEO_SEGMENT_DIR = Path(gettempdir()) / "vlm-prak" / "video-segments"
VIDEO_FRAME_RATE = 5
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoClipSegment:
    path: Path
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class VideoFrame:
    path: Path
    timestamp_seconds: float


def split_and_resize_video(
    video_path: Path | str,
    split_after_seconds: float,
    minimum_segment_length_seconds: float,
    resolution_width: int,
    resolution_height: int,
    output_dir: Path | str = DEFAULT_VIDEO_SEGMENT_DIR,
) -> list[VideoClipSegment]:
    if split_after_seconds <= 0:
        raise ValueError("split_after_seconds must be greater than 0.")
    if minimum_segment_length_seconds < 1:
        raise ValueError("minimum_segment_length_seconds must be at least 1.")
    if resolution_width <= 0 or resolution_height <= 0:
        raise ValueError("Video resolution width and height must be greater than 0.")

    duration_seconds = _probe_duration_seconds(video_path)
    if duration_seconds <= 0:
        raise ValueError(f"Video '{video_path}' has no positive duration.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    segments: list[VideoClipSegment] = []
    split_after_seconds = max(split_after_seconds, minimum_segment_length_seconds)
    for index, (start_seconds, end_seconds) in enumerate(
        _segment_ranges(
            duration_seconds,
            split_after_seconds,
            minimum_segment_length_seconds,
        )
    ):
        target_path = target_dir / _segment_name(video_path, index)
        resize_video(
            video_path,
            target_path,
            resolution_width,
            resolution_height,
            start_seconds=start_seconds,
            duration_seconds=end_seconds - start_seconds,
        )
        segments.append(
            VideoClipSegment(
                path=target_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )

    return segments


def sample_video_frames(
    video_path: Path | str,
    frame_count: int,
    output_dir: Path | str = Path(gettempdir()) / "vlm-prak" / "video-frames",
) -> list[VideoFrame]:
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than 0.")

    duration_seconds = _probe_duration_seconds(video_path)
    if duration_seconds <= 0:
        raise ValueError(f"Video '{video_path}' has no positive duration.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    frames: list[VideoFrame] = []
    for index, timestamp_seconds in enumerate(
        _sample_frame_timestamps(duration_seconds, frame_count)
    ):
        target_path = target_dir / _frame_name(video_path, index, timestamp_seconds)
        extract_video_frame(video_path, target_path, timestamp_seconds)
        frames.append(VideoFrame(path=target_path, timestamp_seconds=timestamp_seconds))
    return frames


def resize_video(
    input_path: Path | str,
    output_path: Path,
    resolution_width: int,
    resolution_height: int,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    frame_rate: int = VIDEO_FRAME_RATE,
    timeout_seconds: float | None = None,
    retry_count: int = 0,
    crf: int | None = None,
) -> Path:
    if frame_rate <= 0:
        raise ValueError("Video frame rate must be greater than 0.")
    if crf is not None and not 0 <= crf <= 51:
        raise ValueError("Video CRF must be between 0 and 51.")

    total_start = timer_start()
    logger.info(
        "timing.ffmpeg.resize.start source_kind=%s source_name=%s "
        "output_name=%s start_seconds=%s duration_seconds=%s fps=%s "
        "resolution=%sx%s timeout_seconds=%s retry_count=%s crf=%s",
        _source_kind(input_path),
        _source_name(input_path),
        output_path.name,
        start_seconds,
        duration_seconds,
        frame_rate,
        resolution_width,
        resolution_height,
        timeout_seconds,
        retry_count,
        crf,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y"]
    if start_seconds is not None:
        command.extend(["-ss", _ffmpeg_seconds(start_seconds)])
    command.extend(["-i", _source_value(input_path)])
    if duration_seconds is not None:
        command.extend(["-t", _ffmpeg_seconds(duration_seconds)])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-vf",
            _scale_filter(resolution_width, resolution_height),
            "-r",
            str(frame_rate),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            *([] if crf is None else ["-crf", str(crf)]),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    _run_command(
        command,
        timeout_seconds=timeout_seconds,
        retry_count=retry_count,
    )
    logger.info(
        "timing.ffmpeg.resize.done source_kind=%s source_name=%s "
        "output_name=%s output_bytes=%s total_ms=%.3f",
        _source_kind(input_path),
        _source_name(input_path),
        output_path.name,
        _file_size(output_path),
        elapsed_ms(total_start),
    )
    return output_path


def extract_video_frame(
    input_path: Path | str,
    output_path: Path,
    timestamp_seconds: float,
) -> Path:
    """Extract one frame from a local or remote video at an exact timestamp."""

    if timestamp_seconds < 0:
        raise ValueError("Video frame timestamp must not be negative.")

    total_start = timer_start()
    logger.info(
        "timing.ffmpeg.frame.start source_kind=%s source_name=%s "
        "output_name=%s timestamp_seconds=%.3f",
        _source_kind(input_path),
        _source_name(input_path),
        output_path.name,
        timestamp_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        _ffmpeg_seconds(timestamp_seconds),
        "-i",
        _source_value(input_path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-an",
        "-q:v",
        "2",
        str(output_path),
    ]
    _run_command(command)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Video frame extraction produced no image at {timestamp_seconds:.3f}s."
        )
    logger.info(
        "timing.ffmpeg.frame.done source_kind=%s source_name=%s "
        "output_name=%s output_bytes=%s total_ms=%.3f",
        _source_kind(input_path),
        _source_name(input_path),
        output_path.name,
        _file_size(output_path),
        elapsed_ms(total_start),
    )
    return output_path


def _probe_duration_seconds(video_path: Path | str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        _source_value(video_path),
    ]
    result = _run_command(command)
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"Could not read duration for video '{video_path}'."
        ) from exc


def _run_command(
    command: list[str],
    *,
    timeout_seconds: float | None = None,
    retry_count: int = 0,
) -> subprocess.CompletedProcess[str]:
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("Video processing timeout must be greater than 0.")
    if retry_count < 0:
        raise ValueError("Video processing retry count must not be negative.")

    executable = command[0] if command else "unknown"
    attempt_count = retry_count + 1
    last_error: subprocess.TimeoutExpired | subprocess.CalledProcessError | None = None
    for attempt in range(1, attempt_count + 1):
        command_start = timer_start()
        logger.info(
            "timing.media_process.start executable=%s attempt=%s "
            "attempt_count=%s timeout_seconds=%s",
            executable,
            attempt,
            attempt_count,
            timeout_seconds,
        )
        try:
            run_arguments = {
                "check": True,
                "capture_output": True,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
            }
            if timeout_seconds is not None:
                run_arguments["timeout"] = timeout_seconds
            result = subprocess.run(
                command,
                **run_arguments,
            )
            logger.info(
                "timing.media_process.done executable=%s attempt=%s "
                "duration_ms=%.3f stdout_chars=%s stderr_chars=%s",
                executable,
                attempt,
                elapsed_ms(command_start),
                len(result.stdout),
                len(result.stderr),
            )
            return result
        except FileNotFoundError as exc:
            logger.exception(
                "timing.media_process.failed executable=%s attempt=%s duration_ms=%.3f",
                executable,
                attempt,
                elapsed_ms(command_start),
            )
            raise RuntimeError(
                f"'{executable}' is required for video processing but was not found."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            logger.warning(
                "timing.media_process.timeout executable=%s attempt=%s "
                "attempt_count=%s timeout_seconds=%s duration_ms=%.3f",
                executable,
                attempt,
                attempt_count,
                timeout_seconds,
                elapsed_ms(command_start),
            )
        except subprocess.CalledProcessError as exc:
            last_error = exc
            logger.warning(
                "timing.media_process.failed executable=%s attempt=%s "
                "attempt_count=%s duration_ms=%.3f return_code=%s "
                "stderr_chars=%s",
                executable,
                attempt,
                attempt_count,
                elapsed_ms(command_start),
                exc.returncode,
                len(exc.stderr or ""),
            )

        if attempt < attempt_count:
            logger.warning(
                "timing.media_process.retry executable=%s completed_attempt=%s "
                "next_attempt=%s",
                executable,
                attempt,
                attempt + 1,
            )

    if isinstance(last_error, subprocess.TimeoutExpired):
        reason = f"timed out after {timeout_seconds} seconds"
    elif isinstance(last_error, subprocess.CalledProcessError):
        stderr = last_error.stderr.strip() if last_error.stderr else str(last_error)
        reason = f"exited with code {last_error.returncode}: {stderr}"
    else:
        reason = "unknown processing failure"
    raise RuntimeError(
        f"Video processing command '{executable}' failed after {attempt_count} "
        f"attempts (initial attempt plus {retry_count} retries): {reason}."
    ) from last_error


def _scale_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )


def _segment_ranges(
    duration_seconds: float,
    split_after_seconds: float,
    minimum_segment_length_seconds: float,
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start_seconds = 0.0
    while start_seconds < duration_seconds:
        end_seconds = min(start_seconds + split_after_seconds, duration_seconds)
        remaining_seconds = duration_seconds - end_seconds
        if 0 < remaining_seconds < minimum_segment_length_seconds:
            end_seconds = duration_seconds
        ranges.append((start_seconds, end_seconds))
        start_seconds = end_seconds
    return ranges


def _segment_name(video_path: Path | str, index: int) -> str:
    source = _source_value(video_path)
    source_hash = sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{_source_stem(source)}-{source_hash}-{index:04d}.mp4"


def _frame_name(video_path: Path | str, index: int, timestamp_seconds: float) -> str:
    source = _source_value(video_path)
    source_hash = sha256(source.encode("utf-8")).hexdigest()[:12]
    timestamp_ms = round(timestamp_seconds * 1000)
    return f"{_source_stem(source)}-{source_hash}-{index:04d}-{timestamp_ms}ms.jpg"


def _sample_frame_timestamps(
    duration_seconds: float,
    frame_count: int,
) -> list[float]:
    return [
        min(duration_seconds, duration_seconds * (index + 1) / (frame_count + 1))
        for index in range(frame_count)
    ]


def _source_value(video_path: Path | str) -> str:
    return str(video_path)


def _source_stem(source: str) -> str:
    parsed_source = urlparse(source)
    if parsed_source.scheme in {"http", "https"}:
        return Path(parsed_source.path).stem or "video"
    return Path(source).stem or "video"


def _source_kind(video_path: Path | str) -> str:
    return (
        "remote" if urlparse(str(video_path)).scheme in {"http", "https"} else "local"
    )


def _source_name(video_path: Path | str) -> str:
    source = str(video_path)
    parsed = urlparse(source)
    return Path(parsed.path if parsed.scheme else source).name or "video"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _ffmpeg_seconds(value: float) -> str:
    return f"{value:.6f}"
