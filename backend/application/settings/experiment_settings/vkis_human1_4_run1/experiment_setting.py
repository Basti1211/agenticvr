"""Configuration for experiment settings."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings


class ExperimentSettings(BaseSettings):
    available_datasets: ClassVar[list[str]] = ["mvk", "v3c", "lhe"]
    available_tasks: ClassVar[list[str]] = ["v-kis", "t-kis", "qa", "avs"]


settings = ExperimentSettings()
