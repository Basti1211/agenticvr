from application.types.retrieval.queries import LoadVideoSummary


def format_load_video_summary(
    queryText: str,
    dataset: str,
    candidateCount: int,
) -> LoadVideoSummary:
    """
    Map from LLM outputs to actual prak text query
    """
    queryText = queryText.split()
    queryText = queryText[1:]
    if len(queryText) != 1:
        raise ValueError("Load video summary query must specify exactly one image ID.")
    query = LoadVideoSummary(
        dataset=dataset,
        itemId=queryText[0],
        candidateCount=500,
    )
    return query


def create_load_video_summary_payload(query: LoadVideoSummary) -> dict:
    _validate_item_id(query.itemId)
    return {
        "item_id": query.itemId,
        "dataset": query.dataset,
        "k": query.candidateCount,
    }


def _validate_item_id(item_id: str) -> None:
    parts = item_id.rsplit("_", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "Load video summary item id must be formatted as '<video_id>_<frame_id>'. "
            f"Got: {item_id!r}"
        )
