""" Settings router for handling API endpoints related to application settings. """

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from application.settings.experiment_setting import settings as experiment_settings


router = APIRouter()


@router.get("/datasets", summary="Get available datasets")
async def get_available_datasets():
    """
    Get a list of available datasets.

    Returns:
        A list of available datasets.
    """
    # Example implementation, replace with actual logic to retrieve datasets
    return {"datasets": experiment_settings.available_datasets}


@router.get("/tasks", summary="Get available tasks")
async def get_available_tasks():
    """
    Get a list of available tasks.

    Returns:
        A list of available tasks.
    """
    # Example implementation, replace with actual logic to retrieve tasks
    return {"tasks": experiment_settings.available_tasks}
