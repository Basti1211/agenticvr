#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""API router entrypoint."""


import logging
from urllib.parse import urljoin
from fastapi import APIRouter

from application.modalities.image import download_image
from application.settings.settings import _data_layer_settings
from application.system.vlm.executor import run
from application.types.vlm.prompt import ImageInput, Prompt
from application.system.retrieval.fetch import run_query

from .services.settings import router as settings_router
from .services.tasks import router as tasks_router


router = APIRouter()
logger = logging.getLogger(__name__)

# add settings router
router.include_router(
    settings_router,
    tags=["settings"],
    prefix="/settings"
)

# add tasks router
router.include_router(
    tasks_router,
    tags=["tasks"],
    prefix="/tasks",
)

@router.get("/test")
async def test() -> str:
    """Run a small default prompt against the data layer"""
    #return await run_in_threadpool(_run_test_vlm)
    candidates = await run_query("t test query || test query 2", "MVK")
    logger.debug("test route candidates=%s", candidates)
    return "TEST"


async def _run_test_vlm() -> str:
    identifier = "MVK/HongKong2_Aug2023_0033/HongKong2_Aug2023_0033_125.jpg"
    image_url = _image_url(identifier)
    image_path = download_image(image_url)

    prompt = Prompt(
        system_prompt="You are a friendly chatbot.",
        context_prompt="Describe the image.",
        context_images=[
            ImageInput(
                identifier=identifier,
                file=image_path,
            )
        ],
        context_videos=[],
    )

    return await run("qwen", prompt)


def _image_url(identifier: str) -> str:
    data_layer_settings = _data_layer_settings()
    return urljoin(data_layer_settings["image_path"], identifier.lstrip("/"))
