from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
import re

from application.settings.settings import (
    _data_layer_settings,
    retrieval_candidate_count,
)
from application.system.retrieval.candidate_filters import (
    CandidateFilter,
    active_candidate_filters,
    apply_candidate_filters,
    first_result_per_video,
    unique_candidates,
)
from application.system.retrieval.endpoints import get_endpoint_for_query
from application.system.retrieval.format_candidates import (
    group_temporal_candidates,
    transform_candidates,
)
from application.system.retrieval.proxy import DataLayerProxy
from application.system.retrieval.queries.bayesUpdate import (
    create_bayes_update_payload,
    format_bayes_update,
)
from application.system.retrieval.queries.directAnswer import format_direct_answer
from application.system.retrieval.queries.imageQuery import (
    create_image_query_payload,
    format_image_query,
)
from application.system.retrieval.queries.intraVideoQuery import (
    create_intra_video_query_payload,
    format_intra_video_query,
)
from application.system.retrieval.queries.loadMore import (
    create_load_more_payload,
    format_load_more_query,
)
from application.system.retrieval.queries.loadVideoSummary import (
    create_load_video_summary_payload,
    format_load_video_summary,
)
from application.system.retrieval.queries.historyTraversal import (
    format_history_traversal_query,
)
from application.system.retrieval.queries.temporalQuery import (
    create_temporal_query_payload,
    format_temporal_query,
)
from application.system.retrieval.queries.textQuery import (
    create_text_query_payload,
    format_text_query,
)
from application.types.retrieval.candidate import Candidate
from application.types.retrieval.queries import (
    BayesUpdate,
    DirectAnswer,
    HistoryTraversal,
    ImageQuery,
    IntraVideoQuery,
    LoadMore,
    LoadVideoSummary,
    Query,
    TemporalQuery,
    TextQuery,
)
from fastapi import HTTPException
from pydantic import ValidationError

QueryFormatter = Callable[[str, str, int], Query]
PayloadBuilder = Callable[[Query], dict]
DATA_LAYER_TIMING_KEYS = {"T3_dataServiceReceived", "T4_dataServiceProcessedQuery"}
LOAD_VIDEO_SUMMARY_WINDOW_SIZE = 16
LOAD_VIDEO_SUMMARY_TARGET_PRECEDING_COUNT = 8

QUERY_FORMATTERS: dict[str, QueryFormatter] = {
    "nl": format_text_query,
    "lnl": format_text_query,
    "t": format_temporal_query,
    "b": format_bayes_update,
    "lvs": format_load_video_summary,
    "iv": format_intra_video_query,
    "im": format_image_query,
    "an": format_direct_answer,
    "back": format_history_traversal_query,
}

PAYLOAD_BUILDERS: dict[type[Query], PayloadBuilder] = {
    TextQuery: create_text_query_payload,
    TemporalQuery: create_temporal_query_payload,
    LoadVideoSummary: create_load_video_summary_payload,
    IntraVideoQuery: create_intra_video_query_payload,
    ImageQuery: create_image_query_payload,
    LoadMore: create_load_more_payload,
}


@dataclass
class RetrievalPageCache:
    query: Query
    action: str
    endpoint: str
    payload: dict
    candidates: list[Candidate]
    next_offset: int
    requested_count: int


@dataclass
class RetrievalHistorySnapshot:
    cache: RetrievalPageCache
    result: list[Candidate]


_LOAD_MORE_CACHE: dict[str, RetrievalPageCache] = {}
_HISTORY: dict[str, list[RetrievalHistorySnapshot]] = {}
_HISTORY_CURSOR: dict[str, int] = {}


async def run_query(query: str, dataset: str) -> list[Candidate]:
    """Fetch and page candidates from the data layer."""

    query_obj = evaluate_query(query, dataset)
    if isinstance(query_obj, LoadMore):
        return await load_more(query_obj)
    if isinstance(query_obj, HistoryTraversal):
        return traverse_history(query_obj)
    if isinstance(query_obj, BayesUpdate):
        return await bayes_update(query_obj, query)

    endpoint = get_endpoint_for_query(query_obj)
    payload = create_payload(query_obj)
    candidates = await fetch_candidates_for_query(query_obj, endpoint, payload)

    page_size = _page_size()
    result = (
        candidates
        if isinstance(query_obj, LoadVideoSummary)
        else candidates[:page_size]
    )
    cache = RetrievalPageCache(
        query=query_obj,
        action=query,
        endpoint=endpoint,
        payload=payload,
        candidates=candidates,
        next_offset=len(result),
        requested_count=_candidate_count(query_obj),
    )
    _LOAD_MORE_CACHE[_cache_key(dataset)] = cache

    _record_history(dataset, cache, result)
    return result


async def run_query_for_browser(
    query: str,
    dataset: str,
    seen_candidate_ids: set[str],
    candidate_limit: int,
) -> list[Candidate]:
    """Return one ranking window, optionally excluding task-scoped seen IDs."""

    if candidate_limit < 1:
        raise ValueError("Browser candidate limit must be at least 1.")

    page_size = min(_page_size(), candidate_limit)
    query_obj = evaluate_query(query, dataset)
    if isinstance(query_obj, LoadMore):
        return await load_more_for_browser(
            query_obj,
            seen_candidate_ids,
            page_size,
        )
    if isinstance(query_obj, HistoryTraversal):
        return await traverse_history_for_browser(
            query_obj,
            seen_candidate_ids,
            page_size,
        )
    if isinstance(query_obj, BayesUpdate):
        cache = await _bayes_ranking(query_obj, query)
    else:
        cache = await _query_ranking(query_obj, query, dataset)

    if isinstance(query_obj, LoadVideoSummary):
        return _complete_unpaged_browser_result(
            dataset,
            cache,
            seen_candidate_ids if _unseen(query_obj) else None,
        )

    if _unseen(query_obj):
        return await _next_new_candidate_window(
            dataset,
            cache,
            seen_candidate_ids,
            page_size,
        )
    return await _next_browser_ranking_window(
        dataset,
        cache,
        page_size,
    )


async def bayes_update(query: BayesUpdate, action: str) -> list[Candidate]:
    cache = _LOAD_MORE_CACHE.get(_cache_key(query.dataset))
    if cache is None:
        raise ValueError("Bayes Update requested before any retrieval query.")

    endpoint = get_endpoint_for_query(query)
    payload = create_bayes_update_payload(query, cache.candidates)
    candidates = await fetch_candidates_for_query(query, endpoint, payload)

    page_size = _page_size()
    cache = RetrievalPageCache(
        query=query,
        action=action,
        endpoint=endpoint,
        payload=payload,
        candidates=candidates,
        next_offset=min(page_size, len(candidates)),
        requested_count=_candidate_count(query),
    )
    _LOAD_MORE_CACHE[_cache_key(query.dataset)] = cache

    result = candidates[:page_size]
    _record_history(query.dataset, cache, result)
    return result


async def load_more_for_browser(
    query: LoadMore,
    seen_candidate_ids: set[str],
    page_size: int,
) -> list[Candidate]:
    """Return the next fixed-size page for browser inspection."""

    cache = _LOAD_MORE_CACHE.get(_cache_key(query.dataset))
    if cache is None:
        raise ValueError("Load More requested before any retrieval query.")
    if isinstance(cache.query, LoadVideoSummary):
        raise ValueError("Load More cannot follow Load Video Summary.")

    options_changed = _apply_load_more_options(cache, query)
    if options_changed or _unseen(query):
        return await _next_new_candidate_window(
            query.dataset,
            cache,
            seen_candidate_ids,
            page_size,
        )
    return await _next_browser_ranking_window(
        query.dataset,
        cache,
        page_size,
    )


async def load_more(query: LoadMore) -> list[Candidate]:
    cache = _LOAD_MORE_CACHE.get(_cache_key(query.dataset))
    if cache is None:
        raise ValueError("Load More requested before any retrieval query.")
    if isinstance(cache.query, LoadVideoSummary):
        raise ValueError("Load More cannot follow Load Video Summary.")

    page_size = query.candidateCount
    options_changed = _apply_load_more_options(cache, query)
    if options_changed:
        return await _next_new_candidate_window(
            query.dataset,
            cache,
            _history_candidate_ids(query.dataset),
            page_size,
        )

    next_offset = cache.next_offset
    if next_offset < len(cache.candidates):
        cache.next_offset = min(next_offset + page_size, len(cache.candidates))
        result = cache.candidates[next_offset:cache.next_offset]
        _record_history(query.dataset, cache, result)
        return result

    expanded_query = _with_candidate_count(
        cache.query,
        max(cache.requested_count * 2, 1000),
    )
    endpoint = get_endpoint_for_query(expanded_query)
    payload = dict(cache.payload)
    payload["k"] = _candidate_count(expanded_query)
    candidates = await fetch_candidates_for_query(expanded_query, endpoint, payload)

    cache.query = expanded_query
    cache.endpoint = endpoint
    cache.payload = payload
    cache.candidates = candidates
    cache.requested_count = _candidate_count(expanded_query)
    cache.next_offset = min(next_offset + page_size, len(candidates))
    result = candidates[next_offset:cache.next_offset]
    _record_history(query.dataset, cache, result)
    return result


def traverse_history(query: HistoryTraversal) -> list[Candidate]:
    restored_cache, restored_result = _restore_history_snapshot(query)
    _LOAD_MORE_CACHE[_cache_key(query.dataset)] = restored_cache
    _record_history(query.dataset, restored_cache, restored_result)
    return restored_result


async def traverse_history_for_browser(
    query: HistoryTraversal,
    seen_candidate_ids: set[str],
    page_size: int,
) -> list[Candidate]:
    """Restore the exact prior ranking page, including previously seen candidates."""

    return traverse_history(query)


async def fetch_candidates_for_query(
    query: Query,
    endpoint: str,
    payload: dict,
) -> list[Candidate]:
    if isinstance(query, LoadMore):
        raise ValueError("Load More is served from the local retrieval cache.")
    if isinstance(query, HistoryTraversal):
        raise ValueError("History traversal is served from the local retrieval cache.")
    candidates = await fetch_candidates(endpoint, payload)
    if isinstance(query, TemporalQuery):
        candidates = group_temporal_candidates(candidates)
    return apply_candidate_filters(
        candidates,
        [*_candidate_filters(query), *active_candidate_filters()],
    )


async def fetch_candidates(
    endpoint: str,
    payload: dict,
) -> list[Candidate]:
    proxy = DataLayerProxy()
    response = await proxy.post_search(endpoint, payload)
    if not isinstance(response, list):
        raise HTTPException(status_code=502, detail="Invalid response format")

    # The data layer appends service timing metadata after the candidate records.
    raw_candidates = [
        item
        for item in response
        if not _is_data_layer_timing(item)
    ]
    try:
        candidates = transform_candidates(raw_candidates)
    except (ValidationError, ValueError) as e:
        raise HTTPException(status_code=502, detail="Invalid candidate response") from e

    return candidates


def evaluate_query(query: str, dataset: str) -> Query:
    """Parse a planner action into a typed retrieval query."""

    query, one_result_per_video, unseen = _extract_query_options(query)
    parts = query.split()
    if not parts:
        raise ValueError("Query is empty.")

    first_token = _query_token(parts[0])
    if first_token.startswith("back("):
        first_token = "back"
    if first_token in {"lm", "more", "load_more"}:
        query_obj = format_load_more_query(query, dataset, _page_size())
        return _with_query_options(query_obj, one_result_per_video, unseen)

    formatter = QUERY_FORMATTERS.get(first_token)
    if formatter is None:
        raise ValueError(f"Unknown first token: {first_token}")
    query_obj = formatter(query, dataset, _fetch_candidate_count())
    return _with_query_options(query_obj, one_result_per_video, unseen)

    
def create_payload(query: Query) -> dict:
    """Create the data-layer payload for a typed query."""

    if isinstance(query, DirectAnswer):
        raise ValueError("Direct Answer is handled locally by the retrieval loop.")
    if isinstance(query, BayesUpdate):
        raise ValueError("Bayes Update payload requires cached previous query candidates.")
    if isinstance(query, HistoryTraversal):
        raise ValueError("History traversal is handled locally by the retrieval cache.")

    builder = PAYLOAD_BUILDERS.get(type(query))
    if builder is None:
        raise ValueError(f"Unknown query type: {type(query).__name__}")
    return builder(query)


def reset_load_more_cache() -> None:
    _LOAD_MORE_CACHE.clear()
    _HISTORY.clear()
    _HISTORY_CURSOR.clear()


def _page_size() -> int:
    return retrieval_candidate_count(_data_layer_settings())


def _fetch_candidate_count() -> int:
    settings = _data_layer_settings()
    return int(
        settings.get("fetch_candidates_multiplier", 10)
    ) * retrieval_candidate_count(settings)


def _cache_key(dataset: str) -> str:
    return dataset.lower()


async def _query_ranking(
    query: Query,
    action: str,
    dataset: str,
) -> RetrievalPageCache:
    endpoint = get_endpoint_for_query(query)
    payload = create_payload(query)
    candidates = await fetch_candidates_for_query(query, endpoint, payload)
    cache = RetrievalPageCache(
        query=query,
        action=action,
        endpoint=endpoint,
        payload=payload,
        candidates=candidates,
        next_offset=0,
        requested_count=_candidate_count(query),
    )
    _LOAD_MORE_CACHE[_cache_key(dataset)] = cache
    return cache


async def _bayes_ranking(
    query: BayesUpdate,
    action: str,
) -> RetrievalPageCache:
    previous_cache = _LOAD_MORE_CACHE.get(_cache_key(query.dataset))
    if previous_cache is None:
        raise ValueError("Bayes Update requested before any retrieval query.")

    endpoint = get_endpoint_for_query(query)
    payload = create_bayes_update_payload(query, previous_cache.candidates)
    candidates = await fetch_candidates_for_query(query, endpoint, payload)
    cache = RetrievalPageCache(
        query=query,
        action=action,
        endpoint=endpoint,
        payload=payload,
        candidates=candidates,
        next_offset=0,
        requested_count=_candidate_count(query),
    )
    _LOAD_MORE_CACHE[_cache_key(query.dataset)] = cache
    return cache


def _apply_load_more_options(
    cache: RetrievalPageCache,
    query: LoadMore,
) -> bool:
    if query.oneResultPerVideo and not _one_result_per_video(cache.query):
        cache.query = cache.query.model_copy(update={"oneResultPerVideo": True})
        cache.candidates = apply_candidate_filters(
            cache.candidates,
            _candidate_filters(cache.query),
        )
        cache.next_offset = 0
        return True
    return False


async def _next_browser_ranking_window(
    dataset: str,
    cache: RetrievalPageCache,
    page_size: int,
) -> list[Candidate]:
    start_offset = cache.next_offset
    result = cache.candidates[start_offset : start_offset + page_size]
    if len(result) < page_size:
        if not isinstance(cache.query, LoadVideoSummary):
            await _expand_cached_ranking(cache)
            result = cache.candidates[start_offset : start_offset + page_size]

    cache.next_offset = start_offset + len(result)
    _record_history(dataset, cache, result)
    return result


async def _next_new_candidate_window(
    dataset: str,
    cache: RetrievalPageCache,
    seen_candidate_ids: set[str],
    page_size: int,
) -> list[Candidate]:
    start_offset = cache.next_offset
    result, next_offset = _collect_new_candidates(
        cache.candidates,
        start_offset,
        seen_candidate_ids,
        page_size,
    )
    if len(result) < page_size and not isinstance(cache.query, LoadVideoSummary):
        await _expand_cached_ranking(cache)
        result, next_offset = _collect_new_candidates(
            cache.candidates,
            start_offset,
            seen_candidate_ids,
            page_size,
        )

    cache.next_offset = next_offset
    _record_history(dataset, cache, result)
    return result


def _collect_new_candidates(
    candidates: list[Candidate],
    start_offset: int,
    seen_candidate_ids: set[str],
    page_size: int,
) -> tuple[list[Candidate], int]:
    result: list[Candidate] = []
    index = start_offset
    while index < len(candidates) and len(result) < page_size:
        candidate = candidates[index]
        index += 1
        candidate_id = "_".join(candidate.id)
        if candidate_id in seen_candidate_ids:
            continue
        result.append(candidate)
    return result, index


def _history_candidate_ids(dataset: str) -> set[str]:
    return {
        "_".join(candidate.id)
        for snapshot in _HISTORY.get(_cache_key(dataset), [])
        for candidate in snapshot.result
    }


def _complete_unpaged_browser_result(
    dataset: str,
    cache: RetrievalPageCache,
    seen_candidate_ids: set[str] | None = None,
) -> list[Candidate]:
    result = (
        cache.candidates
        if seen_candidate_ids is None
        else [
            candidate
            for candidate in cache.candidates
            if "_".join(candidate.id) not in seen_candidate_ids
        ]
    )
    cache.next_offset = len(result)
    _record_history(dataset, cache, result)
    return result


async def _expand_cached_ranking(cache: RetrievalPageCache) -> None:
    if isinstance(cache.query, LoadVideoSummary):
        raise ValueError("Load Video Summary cannot be expanded as a data-layer query.")

    expanded_query = _with_candidate_count(
        cache.query,
        max(cache.requested_count * 2, 1000),
    )
    endpoint = get_endpoint_for_query(expanded_query)
    payload = dict(cache.payload)
    payload["k"] = _candidate_count(expanded_query)
    candidates = await fetch_candidates_for_query(expanded_query, endpoint, payload)

    cache.query = expanded_query
    cache.endpoint = endpoint
    cache.payload = payload
    cache.candidates = candidates
    cache.requested_count = _candidate_count(expanded_query)


def _restore_history_snapshot(
    query: HistoryTraversal,
) -> tuple[RetrievalPageCache, list[Candidate]]:
    key = _cache_key(query.dataset)
    history = _HISTORY.get(key)
    if not history:
        raise ValueError("History traversal requested before any retrieval query.")

    cursor = _HISTORY_CURSOR.get(key, len(history) - 1)
    if cursor <= 0:
        raise ValueError("Cannot move back because there is no previous history step.")

    target = max(cursor - query.steps, 0)
    snapshot = history[target]
    return deepcopy(snapshot.cache), deepcopy(snapshot.result)


def _query_token(value: str) -> str:
    return value[:-1].lower() if value.endswith(":") else value.lower()


def _extract_query_options(query: str) -> tuple[str, bool, bool]:
    parts = query.split()
    kept_parts: list[str] = []
    one_result_per_video = False
    unseen = False
    for part in parts:
        normalized = part.lower()
        if normalized in {"--one-result-per-video", "--one-per-video"}:
            one_result_per_video = True
            continue
        if normalized == "--unseen":
            unseen = True
            continue
        kept_parts.append(part)
    return " ".join(kept_parts), one_result_per_video, unseen


def _with_query_options(
    query: Query,
    one_result_per_video: bool,
    unseen: bool = False,
) -> Query:
    updates: dict[str, bool] = {}
    if one_result_per_video and hasattr(query, "oneResultPerVideo"):
        updates["oneResultPerVideo"] = True
    if unseen and hasattr(query, "unseen"):
        updates["unseen"] = True
    if updates:
        return query.model_copy(update=updates)
    return query


def _is_data_layer_timing(item: object) -> bool:
    return isinstance(item, dict) and DATA_LAYER_TIMING_KEYS.issubset(item)


def _record_history(
    dataset: str,
    cache: RetrievalPageCache,
    result: list[Candidate],
) -> None:
    key = _cache_key(dataset)
    history = _HISTORY.setdefault(key, [])

    history.append(
        RetrievalHistorySnapshot(
            cache=deepcopy(cache),
            result=deepcopy(result),
        )
    )
    _HISTORY_CURSOR[key] = len(history) - 1


def _candidate_filters(query: Query) -> list[CandidateFilter]:
    filters: list[CandidateFilter] = [unique_candidates]
    if isinstance(query, LoadVideoSummary):
        filters.append(_video_summary_window_filter(query.itemId))
    if _one_result_per_video(query):
        filters.append(first_result_per_video)
    return filters


def _one_result_per_video(query: Query) -> bool:
    return bool(getattr(query, "oneResultPerVideo", False))


def _unseen(query: Query) -> bool:
    return bool(getattr(query, "unseen", False))


def _candidate_count(query: Query) -> int:
    value = getattr(query, "candidateCount", None)
    if value is None:
        raise ValueError(f"Query type {type(query).__name__} cannot be paged.")
    return int(value)


def _with_candidate_count(query: Query, candidate_count: int) -> Query:
    if isinstance(query, LoadMore):
        raise ValueError("Load More cannot be expanded as a data-layer query.")
    if isinstance(query, LoadVideoSummary):
        raise ValueError("Load Video Summary cannot be expanded as a data-layer query.")
    return query.model_copy(update={"candidateCount": candidate_count})


def _video_summary_window_filter(item_id: str) -> CandidateFilter:
    video_id, frame_id = _split_item_id(item_id)

    def summary_window(candidates: Iterable[Candidate]) -> list[Candidate]:
        same_video_candidates = [
            candidate
            for candidate in candidates
            if candidate.id and candidate.id[0] == video_id
        ]

        sorted_candidates = sorted(
            same_video_candidates,
            key=_candidate_frame_sort_key,
        )
        return _candidate_window_around_item(
            sorted_candidates,
            item_id,
            _frame_sort_key(frame_id),
        )

    return summary_window


def _candidate_window_around_item(
    candidates: list[Candidate],
    item_id: str,
    item_frame_key: tuple[int, int, str],
) -> list[Candidate]:
    if len(candidates) <= LOAD_VIDEO_SUMMARY_WINDOW_SIZE:
        return candidates

    target_index = _candidate_index(candidates, item_id)
    if target_index is None:
        target_index = _candidate_insertion_index(candidates, item_frame_key)

    max_start = len(candidates) - LOAD_VIDEO_SUMMARY_WINDOW_SIZE
    start = min(
        max(target_index - LOAD_VIDEO_SUMMARY_TARGET_PRECEDING_COUNT, 0),
        max_start,
    )
    end = start + LOAD_VIDEO_SUMMARY_WINDOW_SIZE

    return candidates[start:end]


def _candidate_index(candidates: list[Candidate], item_id: str) -> int | None:
    for index, candidate in enumerate(candidates):
        if "_".join(candidate.id) == item_id:
            return index
    return None


def _candidate_insertion_index(
    candidates: list[Candidate],
    item_frame_key: tuple[int, int, str],
) -> int:
    for index, candidate in enumerate(candidates):
        if _candidate_frame_sort_key(candidate) >= item_frame_key:
            return index
    return len(candidates) - 1


def _candidate_frame_sort_key(candidate: Candidate) -> tuple[int, int, str]:
    frame_id = candidate.id[-1] if candidate.id else ""
    return _frame_sort_key(frame_id)


def _frame_sort_key(frame_id: str) -> tuple[int, int, str]:
    if frame_id.isdecimal():
        return (0, int(frame_id), frame_id)

    digit_matches = re.findall(r"\d+", frame_id)
    if digit_matches:
        return (0, int(digit_matches[-1]), frame_id)

    return (1, 0, frame_id)


def _split_item_id(item_id: str) -> tuple[str, str]:
    parts = item_id.rsplit("_", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "Load video summary item id must be formatted as '<video_id>_<frame_id>'. "
            f"Got: {item_id!r}"
        )
    return parts[0], parts[1]
