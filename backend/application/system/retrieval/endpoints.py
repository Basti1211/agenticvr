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


def get_endpoint_for_query(query: Query) -> str:
    """Returns the endpoint for a given query type.

    Args:
        query_type (str): The type of the query.
    Returns:
        str: The corresponding endpoint.
    Raises:
        ValueError: If the query type is not recognized.
    """
    if isinstance(query, TextQuery):
        return "textQuery"
    elif isinstance(query, TemporalQuery):
        return "temporalQuery"
    elif isinstance(query, BayesUpdate):
        return "bayes"
    elif isinstance(query, LoadVideoSummary):
        return "getVideoFrames"
    elif isinstance(query, IntraVideoQuery):
        return "intraVideoQuery"
    elif isinstance(query, ImageQuery):
        return "imageQueryByID"
    elif isinstance(query, LoadMore):
        return "loadMore"
    elif isinstance(query, HistoryTraversal):
        return "historyTraversal"
    elif isinstance(query, DirectAnswer):
        return "directAnswer"
    else:
        raise ValueError(f"Unknown query type: {type(query).__name__}")
