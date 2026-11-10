from application.types.retrieval.candidate import Candidate
from application.types.retrieval.queries import BayesUpdate


def format_bayes_update(queryText: str, dataset: str, candidateCount: int) -> BayesUpdate:
    """
    Map from LLM outputs to actual prak text query
    """
    queryText = queryText.split()
    queryText = queryText[1:]
    if len(queryText) == 0:
        raise ValueError("Bayes query is empty.")
    query = BayesUpdate(
        dataset=dataset,
        candidateIds=queryText,
        candidateCount=candidateCount,
    )
    return query


def create_bayes_update_payload(
    query: BayesUpdate,
    previous_candidates: list[Candidate],
) -> dict:
    if not previous_candidates:
        raise ValueError("Bayes Update requires candidates from a previous query.")

    selected_images = [
        _split_candidate_id(candidate_id) for candidate_id in query.candidateIds
    ]
    previous_ids = [candidate.id for candidate in previous_candidates]
    previous_scores = [_candidate_score(candidate) for candidate in previous_candidates]
    previous_id_keys = {_candidate_key(candidate) for candidate in previous_candidates}
    unknown_ids = [
        candidate_id
        for candidate_id in query.candidateIds
        if candidate_id not in previous_id_keys
    ]
    if unknown_ids:
        raise ValueError(
            "Bayes Update selected images must come from the previous query results. "
            f"Unknown ids: {', '.join(unknown_ids)}"
        )

    return {
        "selected_images": selected_images,
        "ids": previous_ids,
        "scores": previous_scores,
        "max_rank": 0,
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


def _candidate_key(candidate: Candidate) -> str:
    return "_".join(candidate.id)


def _candidate_score(candidate: Candidate) -> float:
    score = candidate.score
    if score is None:
        return 0.0
    if isinstance(score, str):
        score = score.strip()
        if score.startswith("[") and score.endswith("]"):
            score = score[1:-1].split(",", 1)[0].strip()
    return float(score)
