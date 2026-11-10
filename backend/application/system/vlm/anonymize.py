"""Helpers for removing semantically meaningful media IDs from VLM inputs."""

from __future__ import annotations

from application.types.vlm.prompt import Prompt


class MediaIdAnonymizer:
    """Assign stable anonymous media IDs within a related set of prompts."""

    def __init__(self) -> None:
        self._task_image_aliases: dict[str, str] = {}
        self._task_video_aliases: dict[str, str] = {}
        self._image_aliases: dict[str, str] = {}
        self._video_aliases: dict[str, str] = {}

    def task_image_alias(self, identifier: str) -> str:
        return self._alias(identifier, self._task_image_aliases, "task_image")

    def image_alias(self, identifier: str) -> str:
        return self._alias(identifier, self._image_aliases, "image")

    def task_video_alias(self, identifier: str) -> str:
        return self._alias(identifier, self._task_video_aliases, "task_video")

    def video_alias(self, identifier: str) -> str:
        return self._alias(identifier, self._video_aliases, "video")

    def anonymize_prompt(self, prompt: Prompt) -> Prompt:
        return prompt.model_copy(
            update={
                "task_images": [
                    image.model_copy(
                        update={
                            "identifier": self.task_image_alias(image.identifier)
                        }
                    )
                    for image in prompt.task_images
                ],
                "task_videos": [
                    video.model_copy(
                        update={"identifier": self.task_video_alias(video.identifier)}
                    )
                    for video in prompt.task_videos
                ],
                "context_images": [
                    image.model_copy(
                        update={"identifier": self.image_alias(image.identifier)}
                    )
                    for image in prompt.context_images
                ],
                "context_videos": [
                    video.model_copy(
                        update={"identifier": self.video_alias(video.identifier)}
                    )
                    for video in prompt.context_videos
                ],
            }
        )

    def anonymize_prompts(self, prompts: list[Prompt]) -> list[Prompt]:
        return [self.anonymize_prompt(prompt) for prompt in prompts]

    @staticmethod
    def _alias(identifier: str, aliases: dict[str, str], prefix: str) -> str:
        alias = aliases.get(identifier)
        if alias is None:
            alias = f"{prefix}{len(aliases) + 1}"
            aliases[identifier] = alias
        return alias


def anonymize_prompt_media_ids(prompt: Prompt) -> Prompt:
    return MediaIdAnonymizer().anonymize_prompt(prompt)


def anonymize_prompt_media_ids_batch(prompts: list[Prompt]) -> list[Prompt]:
    return MediaIdAnonymizer().anonymize_prompts(prompts)
