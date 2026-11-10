"""Shared visual and temporal evidence labels for retrieval components."""

from enum import StrEnum


class VisualSimilarity(StrEnum):
    """How closely candidate appearance matches the target appearance."""

    EXACT = "exact"
    VERY_HIGH = "very_high"
    PARTIAL = "partial"
    LOW = "low"
    UNKNOWN = "unknown"


class TemporalStatus(StrEnum):
    """Whether supplied temporal evidence establishes the target sequence."""

    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"
