from pydantic import BaseModel, ConfigDict, Field, RootModel


class CandidateFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(description="URI of one frame in this candidate.")
    id: list[str] = Field(description="Identifiers of this frame.")
    time_keyframe: int = Field(description="Frame timestamp.")
    time_scene_start: int = Field(description="Scene start timestamp.")
    time_scene_end: int = Field(description="Scene end timestamp.")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(description="URI of the candidate image.")
    rank: str | None = Field(
        default=None,
        description="Rank of the candidate in the query result, when applicable.",
    )
    score: str | None = Field(
        default=None,
        description="Similarity score assigned to the candidate, when applicable.",
    )
    id: list[str] = Field(description="Identifiers of the candidate image.")
    time_keyframe: int = Field(description="Keyframe timestamp.")
    time_scene_start: int = Field(description="Scene start timestamp.")
    time_scene_end: int = Field(description="Scene end timestamp.")
    frames: list[CandidateFrame] = Field(
        default_factory=list,
        description=(
            "Ordered keyframes that form one logical candidate. Empty for ordinary "
            "single-frame candidates."
        ),
    )


class Candidates(RootModel[list[Candidate]]):
    pass
