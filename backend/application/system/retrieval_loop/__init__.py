"""Automated retrieval loop package."""

from application.system.retrieval_loop.candidates import (
    CandidateVideoResolver,
    candidate_to_browser_candidate,
)
from application.system.retrieval_loop.loop import (
    DEFAULT_MAX_CANDIDATES_PER_ITERATION,
    RetrievalLoop,
    default_retrieval_loop,
)

__all__ = [
    "RetrievalLoop",
    "DEFAULT_MAX_CANDIDATES_PER_ITERATION",
    "CandidateVideoResolver",
    "candidate_to_browser_candidate",
    "default_retrieval_loop",
]
