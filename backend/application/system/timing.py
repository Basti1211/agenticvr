"""Small helpers for structured runtime timings."""

from __future__ import annotations

import logging
from time import perf_counter

from application.settings.settings import _logging_settings


TIMING_LOG_PREFIX = "timing."
SUMMARY_INFO_EVENTS = frozenset(
    {"retrieval_loop.iteration", "retrieval_loop.answer"}
)


class TimingLogFilter(logging.Filter):
    """Hide detailed INFO records unless timing logging is enabled.

    Retrieval-loop progress and its final outcome remain visible so a normal run
    still reports which iteration is executing and whether an answer was found.
    Warnings and errors are never hidden by this filter.
    """

    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        if self.enabled or record.levelno != logging.INFO:
            return True
        event = record.getMessage().partition(" ")[0]
        return event in SUMMARY_INFO_EVENTS


def timing_logs_enabled() -> bool:
    enabled = _logging_settings().get("timing_logs_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("logging timing_logs_enabled must be a boolean.")
    return enabled


def add_timing_log_filter(handler: logging.Handler) -> bool:
    """Attach the configured detailed-INFO filter and return its enabled state."""

    enabled = timing_logs_enabled()
    for existing_filter in list(handler.filters):
        if isinstance(existing_filter, TimingLogFilter):
            handler.removeFilter(existing_filter)
    handler.addFilter(TimingLogFilter(enabled))
    return enabled


def timer_start() -> float:
    """Return a monotonic timestamp for duration measurements."""

    return perf_counter()


def elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds rounded for API/debug output."""

    return round((perf_counter() - start) * 1000, 3)
