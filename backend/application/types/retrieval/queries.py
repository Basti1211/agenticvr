from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal, TypeAlias


class TextQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(description="The dataset the query is executed on.")
    query: str = Field(description="Text used for a text query.")
    candidateCount: int = Field(description="Number of candidates to return for the query.")
    oneResultPerVideo: bool = Field(
        default=False,
        description="Whether to keep at most one result per source video.",
    )
    unseen: bool = Field(
        default=False,
        description="Whether to return only keyframes not previously displayed in the task.",
    )


class TemporalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(description="The dataset the query is executed on.")
    query1: str = Field(description="First text query.")
    query2: str = Field(description="Second text query.")
    candidateCount: int = Field(description="Number of candidates to return for the query.")
    oneResultPerVideo: bool = Field(
        default=False,
        description="Whether to keep at most one result per source video.",
    )
    unseen: bool = Field(
        default=False,
        description="Whether to return only keyframes not previously displayed in the task.",
    )


class BayesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(description="The dataset the query is executed on.")
    candidateIds: List[str] = Field(description="List of positive candidate IDs.")
    candidateCount: int = Field(description="Number of candidates to return for the query.")
    oneResultPerVideo: bool = Field(
        default=False,
        description="Whether to keep at most one result per source video.",
    )
    unseen: bool = Field(
        default=False,
        description="Whether to return only keyframes not previously displayed in the task.",
    )


class LoadVideoSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    itemId: str = Field(description="The image/keyframe ID that anchors the video summary.")
    dataset: str = Field(description="The dataset the query is executed on.")
    candidateCount: int = Field(description="Number of candidates to return for the query.")
    oneResultPerVideo: bool = Field(
        default=False,
        description="Whether to keep at most one result per source video.",
    )
    unseen: bool = Field(
        default=False,
        description="Whether to return only keyframes not previously displayed in the task.",
    )

class IntraVideoQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(description="The dataset the query is executed on.")
    videoId: str = Field(description="The ID of the video to query.")
    query: str = Field(description="Text used for a text query.")
    candidateCount: int = Field(description="Number of candidates to return for the query.")
    oneResultPerVideo: bool = Field(
        default=False,
        description="Whether to keep at most one result per source video.",
    )
    unseen: bool = Field(
        default=False,
        description="Whether to return only keyframes not previously displayed in the task.",
    )


class ImageQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(description="The dataset the query is executed on.")
    candidateId: str = Field(description="The ID of the image to query.")
    candidateCount: int = Field(description="Number of candidates to return for the query.")
    oneResultPerVideo: bool = Field(
        default=False,
        description="Whether to keep at most one result per source video.",
    )
    unseen: bool = Field(
        default=False,
        description="Whether to return only keyframes not previously displayed in the task.",
    )


class LoadMore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(description="The dataset the query is executed on.")
    candidateCount: int = Field(description="Number of candidates to return for the query.")
    oneResultPerVideo: bool = Field(
        default=False,
        description="Whether to keep at most one result per source video.",
    )
    unseen: bool = Field(
        default=False,
        description="Whether to return only keyframes not previously displayed in the task.",
    )


class HistoryTraversal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(description="The dataset whose action history is traversed.")
    direction: Literal["back"] = Field(
        description="Direction to move through the action history."
    )
    steps: int = Field(
        ge=1,
        description="Number of history steps to traverse.",
    )


class DirectAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(description="The dataset containing the answer video.")
    videoId: str = Field(min_length=1, description="The video ID to submit.")
    timestampMs: int = Field(
        ge=0,
        description="The submitted frame timestamp in milliseconds.",
    )


Query: TypeAlias = (
    TextQuery
    | TemporalQuery
    | BayesUpdate
    | LoadVideoSummary
    | IntraVideoQuery
    | ImageQuery
    | LoadMore
    | HistoryTraversal
    | DirectAnswer
)
