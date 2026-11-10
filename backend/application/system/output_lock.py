"""Per-output locks for concurrent generation of deterministic cache files."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock


_OUTPUT_LOCKS_GUARD = Lock()
_OUTPUT_LOCKS: dict[Path, tuple[Lock, int]] = {}


@contextmanager
def output_lock(output_path: Path) -> Iterator[None]:
    """Serialize one output path while allowing unrelated outputs in parallel."""

    with _OUTPUT_LOCKS_GUARD:
        path_lock, user_count = _OUTPUT_LOCKS.get(output_path, (Lock(), 0))
        _OUTPUT_LOCKS[output_path] = (path_lock, user_count + 1)

    path_lock.acquire()
    try:
        yield
    finally:
        path_lock.release()
        with _OUTPUT_LOCKS_GUARD:
            current_lock, user_count = _OUTPUT_LOCKS[output_path]
            if user_count == 1:
                del _OUTPUT_LOCKS[output_path]
            else:
                _OUTPUT_LOCKS[output_path] = (current_lock, user_count - 1)
