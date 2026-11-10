from application.types.retrieval.queries import TemporalQuery

def format_temporal_query(queryText: str, dataset: str, candidateCount: int) -> TemporalQuery:
    """
    Map from LLM outputs to actual prak text query
    """
    queryText = queryText.split()
    queryText = queryText[1:]
    queryText = " ".join(queryText)
    split_time = queryText.split("||")
    
    if len(split_time) != 2:
        raise ValueError("Temporal query must contain exactly two queries separated by '||'.")
    
    query1, query2 = split_time
    
    query = TemporalQuery(
        dataset=dataset,
        query1=query1.strip(),
        query2=query2.strip(),
        candidateCount=candidateCount,
    )
    return query


def create_temporal_query_payload(query: TemporalQuery) -> dict:
    """
    Creates a payload for the given TemporalQuery object.
    """
    payload = {
        "queries": query.query1,
        "queries2": query.query2,
        "dataset": query.dataset,
        "k": query.candidateCount,
    }
    return payload
