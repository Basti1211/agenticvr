"""Run-local aliases for candidate video identifiers exposed to VLMs."""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from application.types.vlm.prompt import ImageInput, Prompt, VideoInput


ALIAS_SPACE_SIZE = 100_000
_ID_KEYS = {
    "candidate_id",
    "candidate_ids",
    "image_id",
    "keyframe_id",
    "representative_candidate_id",
    "source_candidate_id",
    "source_video_id",
    "video_id",
    "video_ids",
}
_ACTION_WITH_IDENTIFIER = re.compile(
    r"(?i)\b(?P<action>AN|iv|lvs|im|b|IQA|VQA)(?P<spacing>\s+)"
    r"(?P<identifier>[^\s\"'}]+)"
)
_PLAIN_ANSWER = re.compile(
    r"^(?P<leading>\s*)(?P<identifier>[^,\s]+)(?P<separator>\s*,)"
)


class CandidateVideoAliases:
    """Assign stable, unique five-digit aliases during one retrieval run."""

    def __init__(
        self,
        randbelow: Callable[[int], int] = secrets.randbelow,
    ) -> None:
        self._randbelow = randbelow
        self._original_to_alias: dict[str, str] = {}
        self._alias_to_original: dict[str, str] = {}

    def register(self, video_id: str) -> str:
        """Register one original video id and return its stable alias."""

        original = video_id.strip()
        if not original:
            raise ValueError("Candidate video identifier must not be empty.")
        existing = self._original_to_alias.get(original)
        if existing is not None:
            return existing
        if len(self._alias_to_original) >= ALIAS_SPACE_SIZE:
            raise ValueError("The five-digit candidate video alias space is exhausted.")

        while True:
            value = self._randbelow(ALIAS_SPACE_SIZE)
            if not 0 <= value < ALIAS_SPACE_SIZE:
                raise ValueError(
                    "Candidate video alias generator returned an invalid value."
                )
            alias = f"{value:05d}"
            if alias in self._alias_to_original or alias == original:
                continue
            self._original_to_alias[original] = alias
            self._alias_to_original[alias] = original
            return alias

    def anonymize_prompt(self, prompt: Prompt) -> Prompt:
        """Return a model-facing copy without changing the log-facing prompt."""

        if not self._original_to_alias:
            return prompt
        return prompt.model_copy(
            update={
                "context_prompt": self._replace_context_identifiers(
                    prompt.context_prompt,
                    self._original_to_alias,
                ),
                "context_images": [
                    self._anonymize_image(image) for image in prompt.context_images
                ],
                "context_videos": [
                    self._anonymize_video(video) for video in prompt.context_videos
                ],
            }
        )

    def deanonymize_response(self, response: str) -> str:
        """Restore original identifiers before parsing, persistence, or logging."""

        if not self._alias_to_original:
            return response
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            return self._deanonymize_unstructured_response(response)
        restored = self._deanonymize_json_value(payload)
        return json.dumps(restored, ensure_ascii=False, separators=(",", ":"))

    @property
    def original_to_alias(self) -> dict[str, str]:
        """Return a defensive copy for diagnostics and focused tests."""

        return dict(self._original_to_alias)

    def _anonymize_image(self, image: ImageInput) -> ImageInput:
        return image.model_copy(
            update={
                "identifier": self._replace_identifier(
                    image.identifier,
                    self._original_to_alias,
                )
            }
        )

    def _anonymize_video(self, video: VideoInput) -> VideoInput:
        return video.model_copy(
            update={
                "identifier": self._replace_identifier(
                    video.identifier,
                    self._original_to_alias,
                )
            }
        )

    def _deanonymize_json_value(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {
                item_key: self._deanonymize_json_value(item, item_key)
                for item_key, item in value.items()
            }
        if isinstance(value, list):
            return [self._deanonymize_json_value(item, key) for item in value]
        if not isinstance(value, str):
            return value
        if key == "action_call":
            restored = self._replace_action_identifier(
                value,
                self._alias_to_original,
            )
            return self._replace_suffixed_identifiers(
                restored,
                self._alias_to_original,
            )
        if key in _ID_KEYS:
            return self._replace_identifier(value, self._alias_to_original)
        return self._replace_suffixed_identifiers(value, self._alias_to_original)

    def _deanonymize_unstructured_response(self, value: str) -> str:
        restored = self._replace_action_identifier(value, self._alias_to_original)
        restored = self._replace_suffixed_identifiers(
            restored,
            self._alias_to_original,
        )
        plain_answer = _PLAIN_ANSWER.match(restored)
        if plain_answer is None:
            return restored
        identifier = self._replace_identifier(
            plain_answer.group("identifier"),
            self._alias_to_original,
        )
        return (
            f"{plain_answer.group('leading')}{identifier}"
            f"{plain_answer.group('separator')}{restored[plain_answer.end() :]}"
        )

    @classmethod
    def _replace_context_identifiers(
        cls,
        value: str,
        replacements: dict[str, str],
    ) -> str:
        if not replacements:
            return value
        alternatives = cls._alternatives(replacements)
        pattern = re.compile(
            rf"(?P<quoted>(?P<quote>[\"'])(?P<quoted_id>{alternatives})(?P=quote))"
            rf"|(?P<suffixed>(?<![A-Za-z0-9<])(?P<suffix_id>{alternatives})(?=[_:]))"
            rf"|(?P<action>(?P<action_head>(?i:AN|iv|lvs|im|b|IQA|VQA)\s+)"
            rf"(?P<action_id>{alternatives})(?=\s|[\"'}}]|$))"
            rf"|(?P<narrative>(?P<narrative_head>\bvideo\s+[\"']?)"
            rf"(?P<narrative_id>{alternatives})(?=[\"'.,\s]|$))"
        )

        def replace(match: re.Match[str]) -> str:
            if (identifier := match.group("quoted_id")) is not None:
                quote = match.group("quote")
                return f"{quote}{replacements[identifier]}{quote}"
            if (identifier := match.group("suffix_id")) is not None:
                return replacements[identifier]
            if (identifier := match.group("action_id")) is not None:
                return f"{match.group('action_head')}{replacements[identifier]}"
            identifier = match.group("narrative_id")
            return f"{match.group('narrative_head')}{replacements[identifier]}"

        return pattern.sub(replace, value)

    @classmethod
    def _replace_action_identifier(
        cls,
        value: str,
        replacements: dict[str, str],
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            identifier = cls._replace_identifier(
                match.group("identifier"),
                replacements,
            )
            return f"{match.group('action')}{match.group('spacing')}{identifier}"

        return _ACTION_WITH_IDENTIFIER.sub(replace, value)

    @classmethod
    def _replace_suffixed_identifiers(
        cls,
        value: str,
        replacements: dict[str, str],
    ) -> str:
        if not replacements:
            return value
        alternatives = cls._alternatives(replacements)
        pattern = re.compile(
            rf"(?<![A-Za-z0-9<])(?P<identifier>{alternatives})(?=[_:])"
        )
        return pattern.sub(
            lambda match: replacements[match.group("identifier")],
            value,
        )

    @staticmethod
    def _replace_identifier(value: str, replacements: dict[str, str]) -> str:
        direct = replacements.get(value)
        if direct is not None:
            return direct
        for identifier in sorted(replacements, key=len, reverse=True):
            if value.startswith(f"{identifier}_") or value.startswith(f"{identifier}:"):
                return f"{replacements[identifier]}{value[len(identifier) :]}"
        return value

    @staticmethod
    def _alternatives(replacements: dict[str, str]) -> str:
        return "|".join(
            re.escape(identifier)
            for identifier in sorted(replacements, key=len, reverse=True)
        )


_ACTIVE_ALIASES: ContextVar[CandidateVideoAliases | None] = ContextVar(
    "candidate_video_aliases",
    default=None,
)


@contextmanager
def candidate_video_alias_scope(
    aliases: CandidateVideoAliases | None = None,
) -> Iterator[CandidateVideoAliases]:
    """Activate a fresh mapping for exactly one retrieval-loop run."""

    active = aliases or CandidateVideoAliases()
    token = _ACTIVE_ALIASES.set(active)
    try:
        yield active
    finally:
        _ACTIVE_ALIASES.reset(token)


def register_candidate_video_id(video_id: str) -> str | None:
    """Register a candidate id when a retrieval run has an active alias scope."""

    aliases = _ACTIVE_ALIASES.get()
    if aliases is None:
        return None
    return aliases.register(video_id)


def anonymize_candidate_video_ids(prompt: Prompt) -> Prompt:
    """Create the outgoing model prompt for the current run."""

    aliases = _ACTIVE_ALIASES.get()
    if aliases is None:
        return prompt
    return aliases.anonymize_prompt(prompt)


def deanonymize_candidate_video_ids(response: str) -> str:
    """Restore model output to internal identifiers for the current run."""

    aliases = _ACTIVE_ALIASES.get()
    if aliases is None:
        return response
    return aliases.deanonymize_response(response)
