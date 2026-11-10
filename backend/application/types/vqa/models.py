"""Models returned and shared by the VQA module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from application.types.vlm.prompt import VideoInput


@dataclass(frozen=True)
class VQASegment:
    video: VideoInput
    start_seconds: float
    end_seconds: float | None
    source_candidate_id: str


@dataclass(frozen=True)
class VQASegmentAttempt:
    index: int
    start_seconds: float
    end_seconds: float | None
    answer: str | None
    raw_response: str
    verified: bool | None = None
    verification_reason: str | None = None


@dataclass(frozen=True)
class VQAResult:
    answerable: bool
    answer: str | None
    reason: str | None
    observations: tuple[VQASegmentAttempt, ...]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        attempts = [
            {
                "index": item.index,
                "start_seconds": item.start_seconds,
                "end_seconds": item.end_seconds,
                "answer": item.answer,
                "raw_response": item.raw_response,
                "verified": item.verified,
                "verification_reason": item.verification_reason,
            }
            for item in self.observations
        ]
        return {
            "answerable": self.answerable,
            "answer": self.answer,
            "reason": self.reason,
            "observations": attempts,
            "segment_attempts": attempts,
            "metadata": self.metadata,
        }


VQAProgressCallback = Callable[[VQAResult], None]
