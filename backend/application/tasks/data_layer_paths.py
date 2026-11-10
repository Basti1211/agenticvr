"""Data-layer path helpers shared by task import and encoding."""

from __future__ import annotations

from application.tasks.models import Dataset


def dataset_media_path(dataset: Dataset) -> str:
    """Return the data-layer media folder for a dataset."""

    if dataset == Dataset.LHE:
        return "VBSLHE"
    return dataset.value.upper()


def normalize_media_uri_path(uri: str) -> str:
    """Normalize data-layer media URI path prefixes."""

    clean_uri = uri.lstrip("/")
    first_part, separator, remainder = clean_uri.partition("/")
    if first_part.lower() == "lhe":
        return f"VBSLHE{separator}{remainder}" if separator else "VBSLHE"
    return clean_uri
