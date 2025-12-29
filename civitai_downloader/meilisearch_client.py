"""Meilisearch client for CivitAI website search.

This is a private (non-public) API used by `civitai.com/search/images`.
We use it to implement Mode 6 reliably without Playwright scrolling.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx


class CivitaiMeilisearchError(RuntimeError):
    """Base error for Meilisearch client failures."""


class CivitaiMeilisearchAuthError(CivitaiMeilisearchError):
    """Raised when the Meilisearch bearer token is missing or invalid."""


class CivitaiMeilisearchRateLimitError(CivitaiMeilisearchError):
    """Raised when Meilisearch rate limiting prevents progress."""


class CivitaiMeilisearchClient:
    """Async client for CivitAI's private Meilisearch endpoint."""

    DEFAULT_ENDPOINT = "https://search-new.civitai.com/multi-search"
    DEFAULT_INDEX_UID = "images_v6"
    DEFAULT_FACETS: List[str] = [
        "aspectRatio",
        "baseModel",
        "createdAtUnix",
        "tagNames",
        "techniqueNames",
        "toolNames",
        "type",
        "user.username",
    ]

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_s: int = 60,
        request_delay_s: float = 0.25,
    ) -> None:
        self.logger = logging.getLogger("CivitaiDownloader")
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.request_delay_s = request_delay_s

        token = bearer_token or os.getenv("CIVITAI_MEILI_BEARER_TOKEN")
        if not token:
            raise CivitaiMeilisearchAuthError(
                "Missing env var CIVITAI_MEILI_BEARER_TOKEN (required for Mode 6 website search)."
            )
        token = token.strip()
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        self._authorization_header = token

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _build_payload(
        self,
        query: str,
        *,
        index_uid: str,
        limit: int,
        offset: int,
        facets: List[str],
        filters: Optional[List[str]],
    ) -> Dict[str, Any]:
        q: Dict[str, Any] = {
            "q": query,
            "indexUid": index_uid,
            "facets": facets,
            "attributesToHighlight": [],
            "highlightPreTag": "__ais-highlight__",
            "highlightPostTag": "__/ais-highlight__",
            "limit": limit,
            "offset": offset,
        }
        if filters:
            q["filter"] = filters

        return {"queries": [q]}

    async def _post_multi_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = await self._get_client()

        headers = {
            "Authorization": self._authorization_header,
            "Content-Type": "application/json",
            # Not strictly required, but matches what the website sends.
            "X-Meilisearch-Client": "Meilisearch instant-meilisearch ; Meilisearch Python (httpx)",
        }

        # A few targeted retries for 429 (rate limiting).
        for attempt in range(1, 6):
            resp = await client.post(self.endpoint, json=payload, headers=headers)

            if resp.status_code == 401:
                raise CivitaiMeilisearchAuthError(
                    "Meilisearch auth failed (401). Check CIVITAI_MEILI_BEARER_TOKEN."
                )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait_s = float(retry_after) if retry_after else min(60.0, 2.0 * attempt)
                except ValueError:
                    wait_s = min(60.0, 2.0 * attempt)

                self.logger.warning(f"Meilisearch rate limited (429). Waiting {wait_s:.1f}s...")
                await asyncio.sleep(wait_s)
                continue

            resp.raise_for_status()
            return resp.json()

        raise CivitaiMeilisearchRateLimitError(
            "Meilisearch requests repeatedly rate limited (429). Try again later or slow down."
        )

    async def iter_search_hits(
        self,
        query: str,
        *,
        limit: int = 51,
        index_uid: str = DEFAULT_INDEX_UID,
        facets: Optional[List[str]] = None,
        filters: Optional[List[str]] = None,
        max_pages: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield raw Meilisearch hits for a query using offset/limit pagination."""

        if limit < 1:
            raise ValueError("limit must be >= 1")

        facets_to_use = facets if facets is not None else list(self.DEFAULT_FACETS)

        offset = 0
        page = 0
        estimated_total: Optional[int] = None

        while True:
            page += 1
            if max_pages is not None and page > max_pages:
                self.logger.warning(f"Reached max_pages={max_pages} for query '{query}'. Stopping.")
                break

            payload = self._build_payload(
                query,
                index_uid=index_uid,
                limit=limit,
                offset=offset,
                facets=facets_to_use,
                filters=filters,
            )

            data = await self._post_multi_search(payload)
            results = data.get("results") or []
            if not results:
                break

            result0 = results[0] if isinstance(results, list) else None
            if not isinstance(result0, dict):
                break

            hits = result0.get("hits") or []
            if not isinstance(hits, list) or not hits:
                break

            if estimated_total is None:
                et = result0.get("estimatedTotalHits")
                if isinstance(et, int):
                    estimated_total = et

            for hit in hits:
                if isinstance(hit, dict):
                    yield hit

            offset += limit
            if estimated_total is not None and offset >= estimated_total:
                break

            if self.request_delay_s > 0:
                await asyncio.sleep(self.request_delay_s)


