import re

from application.types.retrieval.queries import HistoryTraversal


_HISTORY_TRAVERSAL_PATTERN = re.compile(
    r"^(?P<direction>back)(?:\s+|\()(?P<steps>\d+)\)?$",
    re.IGNORECASE,
)


def format_history_traversal_query(
    query_text: str,
    dataset: str,
    candidate_count: int,
) -> HistoryTraversal:
    del candidate_count
    normalized = " ".join(query_text.strip().split())
    match = _HISTORY_TRAVERSAL_PATTERN.match(normalized)
    if match is None:
        raise ValueError(
            "History traversal must use 'back <steps>' or 'back(<steps>)'."
        )

    steps = int(match.group("steps"))
    if steps < 1:
        raise ValueError("History traversal steps must be at least 1.")

    return HistoryTraversal(
        dataset=dataset,
        direction=match.group("direction").lower(),
        steps=steps,
    )
