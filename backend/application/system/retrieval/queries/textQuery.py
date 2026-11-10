from application.types.retrieval.queries import TextQuery

def format_text_query(queryText: str, dataset: str, candidateCount: int) -> TextQuery:
    """
    Map from LLM outputs to actual prak text query
    """
    queryText = queryText.split()
    queryText = queryText[1:]
    if len(queryText) == 0:
        raise ValueError("Text query is empty.")
    queryText = " ".join(queryText)
    queryText = queryText.lstrip('"').rstrip('"').strip()
    if not queryText:
        raise ValueError("Text query is empty.")

    query = TextQuery(
        dataset=dataset,
        query=queryText,
        candidateCount=candidateCount,
    )
    
    return query


def create_text_query_payload(query: TextQuery) -> dict:
    """
    Creates a payload for the given TextQuery object.
    """
    payload = {
        "queries": query.query,
        "dataset": query.dataset,
        "k": query.candidateCount,
    }
    return payload
