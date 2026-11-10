from application.types.retrieval.queries import ImageQuery

def format_image_query(queryText: str, dataset: str, candidateCount: int) -> ImageQuery:
    """
    Map from LLM outputs to actual prak text query
    """
    queryText = queryText.split()
    queryText = queryText[1:]
    if len(queryText) != 1:
        raise ValueError("Image query must specify exactly one image ID.")
    query = ImageQuery(
        dataset=dataset,
        candidateId=queryText[0],
        candidateCount=candidateCount,
    )
    return query


def create_image_query_payload(query: ImageQuery) -> dict:
    video_id, frame_id = _split_candidate_id(query.candidateId)
    return {
        "video_id": video_id,
        "frame_id": frame_id,
        "dataset": query.dataset,
        "k": query.candidateCount,
    }


def _split_candidate_id(candidate_id: str) -> list[str]:
    parts = candidate_id.rsplit("_", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "Candidate id must be formatted as '<video_id>_<frame_id>'. "
            f"Got: {candidate_id!r}"
        )
    return parts
