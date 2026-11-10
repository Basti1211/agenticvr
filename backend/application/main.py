#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains the main backend logic for the web app.
Individual modules contain all the endpoint logic for user requests.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from application.routes import routes
from application.system.retrieval.proxy import DataLayerProxy
from application.system.timing import add_timing_log_filter


def _configure_application_logging() -> bool:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    application_logger = logging.getLogger("application")
    application_logger.setLevel(level)
    if not application_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        application_logger.addHandler(handler)
    timing_enabled = False
    handlers = list(application_logger.handlers)
    handlers.extend(logging.getLogger().handlers)
    for handler in dict.fromkeys(handlers):
        timing_enabled = add_timing_log_filter(handler)
    application_logger.propagate = False
    return timing_enabled


timing_logs_enabled = _configure_application_logging()
logger = logging.getLogger(__name__)
logger.info(
    "application.logging.configured level=%s timing_logs_enabled=%s",
    logging.getLevelName(logger.getEffectiveLevel()),
    timing_logs_enabled,
)


# FastAPI
app = FastAPI(
    title="PraK-LLM",
    description="PraK-LLM backend API.",
    version="1.0.0",
)


# Middleware: CORS for browser HTTP API calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware: GZIP compression
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Include main router
app.include_router(routes.router)


@app.on_event("shutdown")
async def close_data_layer_proxy() -> None:
    """Release pooled retrieval-service connections on backend shutdown."""

    await DataLayerProxy.close_session()
