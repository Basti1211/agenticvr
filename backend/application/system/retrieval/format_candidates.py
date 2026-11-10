from decimal import Decimal, InvalidOperation

from application.settings.settings import _data_layer_settings
from application.types.retrieval.candidate import Candidate, CandidateFrame, Candidates


def _parse_time(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError("Candidate time values must be strings")

    try:
        parsed_value = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Candidate time values must be numeric") from error

    if not parsed_value.is_finite():
        raise ValueError("Candidate time values must be finite")

    rounded_value = parsed_value.to_integral_value()
    if abs(parsed_value - rounded_value) > Decimal("0.001"):
        raise ValueError("Candidate time values must represent integers")

    return int(rounded_value)


def _extract_candidate_times(
    time: object,
    candidate_id: list[str],
) -> tuple[object, object, object]:
    if not isinstance(time, list):
        raise ValueError(f"Candidate time must be a list, got: {time!r}")

    if len(time) == 4:
        expected_candidate_key = "_".join(candidate_id)
        if time[0] != expected_candidate_key:
            raise ValueError(
                "Candidate time key does not match candidate id: "
                f"{time[0]!r} != {expected_candidate_key!r}"
            )
        time = time[1:]

    if len(time) != 3:
        raise ValueError(
            f"Candidate time must contain three timestamps, got: {time!r}"
        )

    return time[0], time[1], time[2]


def _max_scene_duration_ms() -> int | None:
    value = _data_layer_settings().get("max_scene_duration_seconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Data-layer max scene duration must be numeric")

    duration_ms = value * 1000
    if duration_ms <= 0 or not float(duration_ms).is_integer():
        raise ValueError(
            "Data-layer max scene duration must be a positive whole number "
            "of milliseconds"
        )
    return int(duration_ms)


def _cap_scene_times(
    keyframe_ms: int,
    scene_start_ms: int,
    scene_end_ms: int,
    max_duration_ms: int | None,
) -> tuple[int, int]:
    if max_duration_ms is None or scene_end_ms - scene_start_ms <= max_duration_ms:
        return scene_start_ms, scene_end_ms
    if not scene_start_ms <= keyframe_ms <= scene_end_ms:
        raise ValueError("Candidate keyframe must be within its scene interval")

    # Anchor fixed-size buckets at the original scene start. This both includes the
    # keyframe and ensures that another keyframe in the same capped portion gets the
    # exact same bounds, which preserves scene-cache identity downstream.
    relative_keyframe_ms = min(
        keyframe_ms - scene_start_ms,
        scene_end_ms - scene_start_ms - 1,
    )
    bucket_index = relative_keyframe_ms // max_duration_ms
    capped_start_ms = scene_start_ms + bucket_index * max_duration_ms
    capped_end_ms = min(capped_start_ms + max_duration_ms, scene_end_ms)
    return capped_start_ms, capped_end_ms


def transform_candidates(candidates: list[object]) -> list[Candidate]:
    transformed_candidates: list[object] = []
    max_scene_duration_ms = _max_scene_duration_ms()

    for candidate in candidates:
        if not isinstance(candidate, dict):
            transformed_candidates.append(candidate)
            continue

        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, list) or not all(
            isinstance(part, str) for part in candidate_id
        ):
            transformed_candidates.append(candidate)
            continue

        time_keyframe, time_scene_start, time_scene_end = _extract_candidate_times(
            candidate.get("time"),
            candidate_id,
        )

        parsed_keyframe = _parse_time(time_keyframe)
        parsed_scene_start = _parse_time(time_scene_start)
        parsed_scene_end = _parse_time(time_scene_end)
        capped_scene_start, capped_scene_end = _cap_scene_times(
            parsed_keyframe,
            parsed_scene_start,
            parsed_scene_end,
            max_scene_duration_ms,
        )

        transformed_candidates.append(
            {
                "uri": candidate.get("uri"),
                "rank": candidate.get("rank") or "0",
                "score": candidate.get("score") or "0",
                "id": candidate_id,
                "time_keyframe": parsed_keyframe,
                "time_scene_start": capped_scene_start,
                "time_scene_end": capped_scene_end,
            }
        )

    return Candidates.model_validate(transformed_candidates).root


def group_temporal_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse the four returned temporal keyframes into one answer candidate."""

    temporal_group_size = _data_layer_settings().get("temporal_group_size", 4)

    if not candidates:
        return []
    if len(candidates) % temporal_group_size != 0:
        raise ValueError(
            "Temporal candidate responses must contain complete groups of "
            f"{temporal_group_size} keyframes."
        )

    grouped: list[Candidate] = []
    for index in range(0, len(candidates), temporal_group_size):
        frames = candidates[index : index + temporal_group_size]
        first = frames[0]
        frame_models = [
            CandidateFrame(
                uri=frame.uri,
                id=frame.id,
                time_keyframe=frame.time_keyframe,
                time_scene_start=frame.time_scene_start,
                time_scene_end=frame.time_scene_end,
            )
            for frame in frames
        ]
        frame_ids = ["_".join(frame.id) for frame in frames]
        grouped.append(
            Candidate(
                uri=first.uri,
                rank=first.rank,
                score=first.score,
                id=[
                    first.id[0] if first.id else "temporal",
                    "temporal-" + "--".join(frame_ids),
                ],
                time_keyframe=first.time_keyframe,
                time_scene_start=min(frame.time_scene_start for frame in frames),
                time_scene_end=max(frame.time_scene_end for frame in frames),
                frames=frame_models,
            )
        )

    return grouped
