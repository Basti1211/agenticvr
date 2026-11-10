from application.types.retrieval.queries import LoadMore


def format_load_more_query(queryText: str, dataset: str, candidateCount: int) -> LoadMore:
    query_parts = queryText.split()
    if len(query_parts) != 1:
        raise ValueError("Load More query must be exactly 'lm'.")
    return LoadMore(dataset=dataset, candidateCount=candidateCount)


def create_load_more_payload(query: LoadMore) -> dict:
    return {}
