from application.types.retrieval.queries import DirectAnswer


def format_direct_answer(
    query_text: str,
    dataset: str,
    candidate_count: int,
) -> DirectAnswer:
    del candidate_count
    parts = query_text.split()
    arguments = parts[1:]
    if len(arguments) != 2:
        raise ValueError(
            "Direct answer must specify a video ID and timestamp in milliseconds."
        )

    try:
        timestamp_ms = int(arguments[1])
    except ValueError as error:
        raise ValueError(
            "Direct answer timestamp must be an integer number of milliseconds."
        ) from error

    return DirectAnswer(
        dataset=dataset,
        videoId=arguments[0],
        timestampMs=timestamp_ms,
    )
