from application.types.retrieval.queries import IntraVideoQuery
def format_intra_video_query(queryText: str, dataset: str, candidateCount: int) -> IntraVideoQuery:
    """
    Map from LLM outputs to actual prak text query
    """
    queryText = queryText.split()
    queryText = queryText[1:]
    if len(queryText) < 2:
        raise ValueError("Intra-video query must specify at least a video ID and a query.")
    query = IntraVideoQuery(
        dataset=dataset,
        videoId=queryText[0],
        query=" ".join(queryText[1:]),
        candidateCount=candidateCount,
    )
    return query


def create_intra_video_query_payload(query: IntraVideoQuery) -> dict:
    return {
        "queries": query.query,
        "video_id": query.videoId,
        "dataset": query.dataset,
        "k": query.candidateCount,
    }
