import logging
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import gettempdir
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps

from application.system.output_lock import output_lock
from application.system.timing import elapsed_ms, timer_start


DEFAULT_IMAGE_SIZE = (682, 384)
DEFAULT_IMAGE_DIR = Path(gettempdir()) / "vlm-prak" / "images"
logger = logging.getLogger(__name__)


def download_image(
    url: str,
    output_dir: Path | str | None = None,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    timeout: float = 30.0,
) -> Path:
    """Download an image URL, resize it to `size`, and return the local path."""
    return download(url, output_dir=output_dir, size=size, timeout=timeout)


def download(
    url: str,
    output_dir: Path | str | None = None,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    timeout: float = 30.0,
) -> Path:
    """Download an image URL, resize it to `size`, and return the local path."""
    total_start = timer_start()
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError("Image URL must use http or https.")

    logger.info(
        "timing.image.download.start host=%s source_name=%s timeout_seconds=%s",
        parsed_url.hostname or "",
        Path(parsed_url.path).name or "image",
        timeout,
    )
    http_start = timer_start()
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception:
        logger.exception(
            "timing.image.download.failed host=%s source_name=%s http_ms=%.3f",
            parsed_url.hostname or "",
            Path(parsed_url.path).name or "image",
            elapsed_ms(http_start),
        )
        raise
    http_ms = elapsed_ms(http_start)
    logger.info(
        "timing.image.download.http_done host=%s source_name=%s "
        "status_code=%s response_bytes=%s http_ms=%.3f",
        parsed_url.hostname or "",
        Path(parsed_url.path).name or "image",
        response.status_code,
        len(response.content),
        http_ms,
    )

    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_IMAGE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / _downloaded_image_name(url)
    resize_start = timer_start()
    with output_lock(target_path):
        _resize_image(response.content, target_path, size)
    resize_ms = elapsed_ms(resize_start)
    logger.info(
        "timing.image.download.done host=%s source_name=%s output_bytes=%s "
        "resize_ms=%.3f total_ms=%.3f",
        parsed_url.hostname or "",
        Path(parsed_url.path).name or "image",
        _file_size(target_path),
        resize_ms,
        elapsed_ms(total_start),
    )
    return target_path


def _resize_image(content: bytes, target_path: Path, size: tuple[int, int]) -> None:
    with Image.open(BytesIO(content)) as image:
        resized_image = ImageOps.fit(
            image,
            size,
            method=Image.Resampling.LANCZOS,
        )
        resized_image.convert("RGB").save(target_path, format="JPEG")


def _downloaded_image_name(url: str) -> str:
    parsed_url = urlparse(url)
    source_name = Path(parsed_url.path).stem or "image"
    url_hash = sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{source_name}-{url_hash}.jpg"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
