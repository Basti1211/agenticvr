"""Backend storage for task media uploads."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from tempfile import gettempdir
from typing import Literal
from uuid import uuid4

from fastapi import UploadFile


DEFAULT_UPLOAD_DIR = Path(gettempdir()) / "vlm-prak" / "uploads"
UPLOAD_CHUNK_SIZE = 1024 * 1024


async def save_task_media_upload(
    upload: UploadFile,
    output_dir: Path | str = DEFAULT_UPLOAD_DIR,
) -> tuple[Path, int]:
    """Persist an uploaded media file and return its local path and byte size."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = (
        target_dir
        / f"{uuid4().hex}{_safe_media_extension(upload.filename, upload.content_type)}"
    )
    size_bytes = 0

    with target_path.open("wb") as target_file:
        while chunk := await _read_upload_chunk(upload):
            size_bytes += len(chunk)
            target_file.write(chunk)

    return target_path, size_bytes


def infer_media_type(mime_type: str | None) -> Literal["image", "video"] | None:
    """Infer supported task media type from a browser-supplied MIME type."""

    if mime_type is None:
        return None

    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("video/"):
        return "video"
    return None


def public_task_media_name(
    media_type: Literal["image", "video"],
    media_path: Path,
) -> str:
    """Return the neutral media name exposed outside upload storage."""

    suffix = media_path.suffix.lower()
    if not suffix:
        suffix = ".jpg" if media_type == "image" else ".mp4"
    return f"task{suffix}"


async def _read_upload_chunk(upload: UploadFile) -> bytes:
    """Read from the underlying upload stream without relying on wrapper state."""

    return upload.file.read(UPLOAD_CHUNK_SIZE)


def _safe_media_extension(filename: str | None, mime_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    supported_suffixes = {
        ".jpg",
        ".jpeg",
        ".jpe",
        ".png",
        ".webp",
        ".mp4",
        ".mov",
        ".m4v",
        ".webm",
    }
    if suffix not in supported_suffixes:
        suffix = mimetypes.guess_extension(mime_type or "") or suffix

    if suffix in {".jpeg", ".jpe"}:
        return ".jpg"
    if suffix in supported_suffixes:
        return suffix
    return ".bin"
