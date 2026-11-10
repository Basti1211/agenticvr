#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compose application routers.

This module defines the root FastAPI router and includes API endpoints
under `/api/...`.

"""

from fastapi import APIRouter

from . import api

router = APIRouter()

# add api router
router.include_router(
    api.router,
    prefix="/api",
    responses={404: {"description": "Not found"}},
)
