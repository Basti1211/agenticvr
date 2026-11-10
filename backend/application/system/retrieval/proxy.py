from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import orjson
from fastapi import HTTPException

from application.settings.settings import _data_layer_settings

logger = logging.getLogger(__name__)


class DataLayerProxy:
    """Proxy for the external data/search services.

    The retrieval loop may issue several requests in one run. Reusing the aiohttp
    session keeps DNS cache and TCP connections warm instead of rebuilding them
    for every query.
    """

    DEFAULT_SERVICE_URL = "http://acheron.ms.mff.cuni.cz:42032"
    DATA_SERVICE_URL = _data_layer_settings().get(
        "data_service_url",
        DEFAULT_SERVICE_URL,
    ) # default URL for data service (if only endpoint like '/query/' is given)
    SEARCH_SERVICE_URL = _data_layer_settings().get(
        "search_service_url",
        DEFAULT_SERVICE_URL,
    ) # default URL for search service (if only endpoint like '/query/' is given)

    _session: aiohttp.ClientSession | None = None
    _session_loop: asyncio.AbstractEventLoop | None = None

    async def _post_async(self, url: str, payload: dict[str, Any]) -> Any:
        """POST JSON and parse the JSON response with the shared HTTP session."""

        try:
            async with self._client_session().post(url, json=payload) as response:
                body = await response.read()
                if response.status != 200:
                    error_body = body.decode(
                        response.charset or "utf-8",
                        errors="replace",
                    )
                    logger.error(
                        "Service returned HTTP %s from %s: %s",
                        response.status,
                        url,
                        error_body,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="Bad response from service",
                    )
                return self._decode_json(body)
        except HTTPException:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            logger.error("Service request error to %s: %s", url, error)
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def post_search(self, endpoint: str, payload: dict[str, Any]) -> Any:
        """Executes a post request to the Nginx backend service.

        Args:
            endpoint (str): Endpoint of the request (if local, it should end with '/').
            payload (Dict): Payload to send.

        Returns:
            Dict: Response from the service.

        Raises:
            HTTPException: If the request fails or response is invalid.
        """

        url = f"{self.SEARCH_SERVICE_URL.rstrip('/')}/{endpoint.strip('/')}/"
        return await self._post_async(url, payload)

    @classmethod
    async def close_session(cls) -> None:
        """Close the shared session during application shutdown or tests."""

        if cls._session is not None and not cls._session.closed:
            await cls._session.close()
        cls._session = None
        cls._session_loop = None

    @classmethod
    def _client_session(cls) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()
        if (
            cls._session is None
            or cls._session.closed
            or cls._session_loop is not loop
        ):
            connector = aiohttp.TCPConnector(
                limit_per_host=100,
                force_close=False,
                ttl_dns_cache=30,
            )
            cls._session = aiohttp.ClientSession(connector=connector)
            cls._session_loop = loop
        return cls._session

    @staticmethod
    def _decode_json(body: bytes) -> Any:
        try:
            return orjson.loads(body)
        except orjson.JSONDecodeError as error:
            logger.error("Failed to parse service response: %s", error)
            raise HTTPException(
                status_code=502,
                detail="Invalid response format",
            ) from error
