import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from application.types.retrieval.candidate import Candidate


CandidateFilter = Callable[[Iterable[Candidate]], list[Candidate]]
_ACTIVE_CANDIDATE_FILTERS: ContextVar[tuple[CandidateFilter, ...]] = ContextVar(
    "active_candidate_filters",
    default=(),
)
_COMPETITION_YEAR_PATTERN = re.compile(r"\bvbs(?:20)?(?P<year>\d{2})(?=\D|$)", re.I)
_LAST_INTEGER_PATTERN = re.compile(r"(\d+)(?!.*\d)")


@contextmanager
def scoped_candidate_filters(filters: Iterable[CandidateFilter]) -> Iterator[None]:
    """Apply additional filters to data-layer fetches in the current async context."""

    token = _ACTIVE_CANDIDATE_FILTERS.set(tuple(filters))
    try:
        yield
    finally:
        _ACTIVE_CANDIDATE_FILTERS.reset(token)


def active_candidate_filters() -> list[CandidateFilter]:
    """Return retrieval filters scoped to the current task execution."""

    return list(_ACTIVE_CANDIDATE_FILTERS.get())


def apply_candidate_filters(
    candidates: Iterable[Candidate],
    filters: Iterable[CandidateFilter],
) -> list[Candidate]:
    filtered = list(candidates)
    for candidate_filter in filters:
        filtered = candidate_filter(filtered)
    return filtered


def competition_candidate_filters(
    *,
    dataset: str,
    source_task_name: str | None,
    rules: Iterable[dict[str, Any]] = (),
) -> list[CandidateFilter]:
    """Return filters for known competition/dataset subset restrictions."""

    year = _competition_year(source_task_name)
    if year is None:
        return []

    filters: list[CandidateFilter] = []
    normalized_dataset = dataset.lower()
    for rule in rules:
        if _rule_year(rule) != year:
            continue
        if _rule_dataset(rule) != normalized_dataset:
            continue
        allowed_source_video_id_prefixes = _rule_source_video_id_prefixes(rule)
        if allowed_source_video_id_prefixes is not None:
            filters.append(
                source_video_id_prefix_allowlist(allowed_source_video_id_prefixes)
            )
        video_id_range = _rule_video_id_range(rule)
        if video_id_range is not None:
            filters.append(source_video_id_range(*video_id_range))
    return filters


def source_video_id_prefix_allowlist(
    source_video_id_prefixes: Iterable[str],
) -> CandidateFilter:
    """Keep candidates whose source video id starts with an allowed prefix."""

    allowed_prefixes = {
        _normalized_source_video_id(item) for item in source_video_id_prefixes
    }

    def filter_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
        return [
            candidate
            for candidate in candidates
            if _candidate_source_video_id_starts_with(candidate, allowed_prefixes)
        ]

    return filter_candidates


def source_video_id_range(
    min_video_id: int | None = None,
    max_video_id: int | None = None,
) -> CandidateFilter:
    """Keep candidates whose source video number is inside the configured range."""

    def filter_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
        filtered: list[Candidate] = []
        for candidate in candidates:
            video_number = _candidate_source_video_number(candidate)
            if video_number is None:
                filtered.append(candidate)
                continue
            if min_video_id is not None and video_number < min_video_id:
                continue
            if max_video_id is not None and video_number > max_video_id:
                continue
            filtered.append(candidate)
        return filtered

    return filter_candidates


def unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep the first candidate for each browser identifier, preserving rank order."""

    seen_candidate_ids: set[str] = set()
    filtered: list[Candidate] = []
    for candidate in candidates:
        candidate_id = "_".join(candidate.id)
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        filtered.append(candidate)
    return filtered


def first_result_per_video(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep the highest-ranked candidate for each video, preserving result order."""

    seen_video_ids: set[str] = set()
    filtered: list[Candidate] = []
    for candidate in candidates:
        video_id = _candidate_video_id(candidate)
        if video_id in seen_video_ids:
            continue
        seen_video_ids.add(video_id)
        filtered.append(candidate)
    return filtered


def _candidate_video_id(candidate: Candidate) -> str:
    if candidate.id:
        return candidate.id[0]
    return candidate.uri


def _competition_year(source_task_name: str | None) -> int | None:
    if source_task_name is None:
        return None
    match = _COMPETITION_YEAR_PATTERN.search(source_task_name)
    if match is None:
        return None
    return 2000 + int(match.group("year"))


def _rule_year(rule: dict[str, Any]) -> int | None:
    return _optional_int(rule.get("year"))


def _rule_dataset(rule: dict[str, Any]) -> str | None:
    dataset = rule.get("dataset")
    if not isinstance(dataset, str) or not dataset.strip():
        return None
    return dataset.strip().lower()


def _rule_source_video_id_prefixes(rule: dict[str, Any]) -> list[str] | None:
    raw_prefixes = (
        rule.get("allowed_source_video_id_prefixes")
        or rule.get("allowed_video_id_prefixes")
        or rule.get("allowed_source_video_ids")
        or rule.get("allowed_video_ids")
        or rule.get("source_video_ids")
    )
    if raw_prefixes is None:
        return None
    if not isinstance(raw_prefixes, list):
        raise ValueError(
            "Competition candidate filter allowed source video ids must be a list."
        )
    source_video_id_prefixes = [
        str(item)
        for item in raw_prefixes
        if _normalized_source_video_id(str(item))
    ]
    return source_video_id_prefixes or None


def _rule_video_id_range(
    rule: dict[str, Any],
) -> tuple[int | None, int | None] | None:
    min_video_id = _optional_int(rule.get("min_video_id"))
    max_video_id = _optional_int(rule.get("max_video_id"))
    if min_video_id is None and max_video_id is None:
        return None
    if (
        min_video_id is not None
        and max_video_id is not None
        and min_video_id > max_video_id
    ):
        raise ValueError("Competition candidate filter min_video_id exceeds max_video_id.")
    return min_video_id, max_video_id


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Competition candidate filter value must be an integer: {value!r}."
        ) from error


def _candidate_source_video_number(candidate: Candidate) -> int | None:
    if not candidate.id:
        return None
    match = _LAST_INTEGER_PATTERN.search(candidate.id[0])
    if match is None:
        return None
    return int(match.group(1))


def _normalized_candidate_source_video_id(candidate: Candidate) -> str:
    if candidate.id:
        return _normalized_source_video_id(candidate.id[0])
    return _normalized_source_video_id(candidate.uri)


def _candidate_source_video_id_starts_with(
    candidate: Candidate,
    allowed_prefixes: set[str],
) -> bool:
    source_video_id = _normalized_candidate_source_video_id(candidate)
    return any(source_video_id.startswith(prefix) for prefix in allowed_prefixes)


def _normalized_source_video_id(source_video_id: str) -> str:
    normalized = source_video_id.strip().replace("\\", "/").rstrip("/")
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    if normalized.lower().endswith(".mp4"):
        normalized = normalized[:-4]
    return normalized.casefold()
