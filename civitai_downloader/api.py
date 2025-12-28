"""API client for CivitAI endpoints."""

import httpx
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple, AsyncGenerator
from asyncio import Semaphore

from tenacity import (
    retry, stop_after_attempt, wait_random_exponential,
    retry_if_exception, before_sleep_log
)

from .config import (
    BASE_API_URL, MODELS_API_URL, DEFAULT_HEADERS,
    DEFAULT_TIMEOUT, DEFAULT_RETRIES, RETRYABLE_STATUS_CODES
)

# Retry configuration
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException, httpx.ConnectError,
    httpx.RemoteProtocolError, httpx.ReadError, ConnectionResetError
)


def is_retryable_http_status(exception: BaseException) -> bool:
    """Check if exception is a retryable HTTP status error."""
    return (isinstance(exception, httpx.HTTPStatusError) and 
            exception.response.status_code in RETRYABLE_STATUS_CODES)


def should_retry_exception(exception: BaseException) -> bool:
    """Determine if an exception should trigger a retry."""
    return isinstance(exception, RETRYABLE_EXCEPTIONS) or is_retryable_http_status(exception)


class CivitaiAPI:
    """Async client for CivitAI API operations."""
    
    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        semaphore_limit: int = 5,
        retries: int = DEFAULT_RETRIES
    ):
        """Initialize API client.
        
        Args:
            timeout: Request timeout in seconds
            semaphore_limit: Max concurrent requests
            retries: Number of retry attempts
        """
        self.logger = logging.getLogger('CivitaiDownloader')
        self.timeout = timeout
        self.retries = retries
        self.semaphore = Semaphore(semaphore_limit)
        self.headers = DEFAULT_HEADERS.copy()
        
        self._client: Optional[httpx.AsyncClient] = None
        self.failed_urls: List[str] = []
        
        # Rate limiting state
        self._rate_limit_delay = 0.5  # Base delay between requests
        self._rate_limit_until = 0  # Timestamp when rate limit expires
        self._consecutive_429s = 0  # Track consecutive rate limits
    
    async def _wait_for_rate_limit(self) -> None:
        """Wait if we're being rate limited."""
        import time
        now = time.time()
        if now < self._rate_limit_until:
            wait_time = self._rate_limit_until - now
            self.logger.info(f"Rate limit active, waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
    
    def _handle_rate_limit(self) -> float:
        """Handle a 429 response, return wait time."""
        import time
        self._consecutive_429s += 1
        # Progressive backoff: 5s, 15s, 30s, 60s, 120s max
        wait_time = min(120, 5 * (2 ** self._consecutive_429s))
        self._rate_limit_until = time.time() + wait_time
        self.logger.warning(f"Rate limited! Backing off for {wait_time}s (attempt {self._consecutive_429s})")
        return wait_time
    
    def _reset_rate_limit(self) -> None:
        """Reset rate limit tracking after successful request."""
        if self._consecutive_429s > 0:
            self._consecutive_429s = max(0, self._consecutive_429s - 1)
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self.logger.info("HTTP Client closed.")
    
    async def fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch JSON data from URL with retry logic.
        
        Args:
            url: URL to fetch
        
        Returns:
            Parsed JSON data or None on failure
        """
        client = await self.get_client()
        
        for attempt in range(1 + self.retries):
            try:
                async with self.semaphore:
                    response = await client.get(url)
                
                if response.status_code == 404:
                    self.logger.warning(f"Not found (404): {url}")
                    return None
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code in RETRYABLE_STATUS_CODES and attempt < self.retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Retryable status {e.response.status_code}, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                self.logger.error(f"HTTP error for {url}: {e.response.status_code}")
                self.failed_urls.append(url)
                return None
                
            except RETRYABLE_EXCEPTIONS as e:
                if attempt < self.retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Retryable error: {type(e).__name__}, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                self.logger.error(f"Failed after retries: {url} - {e}")
                self.failed_urls.append(url)
                return None
                
            except Exception as e:
                self.logger.error(f"Unexpected error fetching {url}: {e}")
                self.failed_urls.append(url)
                return None
        
        return None
    
    async def fetch_image_bytes(self, url: str) -> Optional[bytes]:
        """Fetch raw image bytes from URL with rate limit handling.
        
        Args:
            url: Image URL
        
        Returns:
            Raw bytes or None on failure
        """
        client = await self.get_client()
        
        for attempt in range(1 + self.retries + 3):  # Extra attempts for rate limiting
            try:
                # Wait for any active rate limit
                await self._wait_for_rate_limit()
                
                async with self.semaphore:
                    # Add small delay between requests to avoid rate limiting
                    await asyncio.sleep(self._rate_limit_delay)
                    response = await client.get(url)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    wait_time = self._handle_rate_limit()
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                self._reset_rate_limit()
                return response.content
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait_time = self._handle_rate_limit()
                    await asyncio.sleep(wait_time)
                    continue
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Failed to fetch image {url}: {e}")
                return None
                
            except Exception as e:
                if attempt < self.retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Failed to fetch image {url}: {e}")
                return None
        
        self.logger.error(f"Failed to fetch image after all attempts: {url}")
        return None
    
    async def search_models_by_tag(
        self,
        tag: str,
        max_pages: int = 500
    ) -> List[Tuple[int, str]]:
        """Search for models tagged with a specific tag.
        
        Args:
            tag: Tag to search for (use spaces, not underscores)
            max_pages: Maximum pages to fetch
        
        Returns:
            List of (model_id, model_name) tuples
        """
        encoded_tag = tag.replace(" ", "%20")
        url: Optional[str] = f"{MODELS_API_URL}?tag={encoded_tag}&nsfw=True"
        
        models_found: List[Tuple[int, str]] = []
        model_ids_seen: set = set()
        visited_urls: set = set()
        page_count = 0
        
        self.logger.info(f"Searching for models with tag: '{tag}' (Max pages: {max_pages})")
        
        while url and page_count < max_pages:
            page_count += 1
            
            if url in visited_urls:
                self.logger.warning(f"Model search loop detected: {url}")
                break
            visited_urls.add(url)
            
            data = await self.fetch_json(url)
            if not data:
                break
            
            items = data.get('items', [])
            metadata = data.get('metadata', {})
            
            if not items:
                if page_count == 1:
                    self.logger.warning(f"No models found for tag '{tag}'")
                break
            
            # Validate first page
            if page_count == 1:
                tag_found = any(
                    tag.lower() in {t.lower() for t in m.get('tags', []) if isinstance(t, str)}
                    for m in items
                )
                if not tag_found:
                    self.logger.warning(f"Tag '{tag}' not found in model tags - possibly invalid")
                    return []
            
            # Process models
            for model in items:
                mid = model.get('id')
                mname = model.get('name') or f"Unnamed_{mid}"
                if isinstance(mid, int) and mid not in model_ids_seen:
                    models_found.append((mid, mname))
                    model_ids_seen.add(mid)
            
            # Next page
            url = metadata.get('nextPage')
            if url:
                await asyncio.sleep(0.5)  # Rate limiting
        
        self.logger.info(f"Found {len(models_found)} models for tag '{tag}' after {page_count} pages")
        return models_found
    
    async def iter_images_by_tag(
        self,
        tag: str,
        nsfw: str = "X"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Iterate over images directly tagged with a specific tag.
        
        This is the DIRECT image search approach - much more comprehensive
        than model-based search.
        
        Args:
            tag: Tag to search for
            nsfw: NSFW level filter
        
        Yields:
            Individual image items from the API
        """
        encoded_tag = tag.replace(" ", "%20")
        url: Optional[str] = f"{BASE_API_URL}?tag={encoded_tag}&nsfw={nsfw}&limit=200"
        
        page_count = 0
        total_items = 0
        visited_cursors: set = set()
        
        self.logger.info(f"Starting direct image tag search for: '{tag}'")
        
        while url:
            page_count += 1
            
            data = await self.fetch_json(url)
            if not data:
                self.logger.warning(f"Failed to fetch page {page_count} for tag '{tag}'")
                break
            
            items = data.get('items', [])
            metadata = data.get('metadata', {})
            
            if not items:
                break
            
            self.logger.info(f"Processing {len(items)} images from tag search page {page_count}")
            
            for item in items:
                total_items += 1
                yield item
            
            # Handle pagination via cursor
            cursor = metadata.get('nextCursor')
            if cursor and cursor not in visited_cursors:
                visited_cursors.add(cursor)
                url = f"{BASE_API_URL}?tag={encoded_tag}&nsfw={nsfw}&limit=200&cursor={cursor}"
                await asyncio.sleep(0.3)  # Rate limiting
            else:
                url = None
        
        self.logger.info(f"Direct tag search completed: {total_items} images from {page_count} pages")
    
    async def iter_images_by_model(
        self,
        model_id: int,
        nsfw: str = "X"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Iterate over images from a specific model.
        
        Args:
            model_id: CivitAI model ID
            nsfw: NSFW level filter
        
        Yields:
            Individual image items from the API
        """
        url: Optional[str] = f"{BASE_API_URL}?modelId={model_id}&nsfw={nsfw}"
        page_count = 0
        
        while url:
            page_count += 1
            self.logger.debug(f"Fetching model {model_id} images, page {page_count}")
            
            data = await self.fetch_json(url)
            if not data:
                break
            
            items = data.get('items', [])
            if not items:
                break
            
            self.logger.info(f"Processing {len(items)} items from page {page_count}")
            
            for item in items:
                yield item
            
            url = data.get('metadata', {}).get('nextPage')
            if url:
                await asyncio.sleep(0.3)
    
    async def iter_images_by_username(
        self,
        username: str,
        nsfw: str = "X"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Iterate over images from a specific user."""
        url: Optional[str] = f"{BASE_API_URL}?username={username}&nsfw={nsfw}&sort=Newest"
        page_count = 0
        
        while url:
            page_count += 1
            data = await self.fetch_json(url)
            if not data:
                break
            
            items = data.get('items', [])
            if not items:
                break
            
            self.logger.info(f"Processing {len(items)} items from page {page_count}")
            
            for item in items:
                yield item
            
            url = data.get('metadata', {}).get('nextPage')
            if url:
                await asyncio.sleep(0.3)
    
    async def iter_images_by_model_version(
        self,
        model_version_id: int,
        nsfw: str = "X"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Iterate over images from a specific model version."""
        url: Optional[str] = f"{BASE_API_URL}?modelVersionId={model_version_id}&nsfw={nsfw}"
        page_count = 0
        
        while url:
            page_count += 1
            data = await self.fetch_json(url)
            if not data:
                break
            
            items = data.get('items', [])
            if not items:
                break
            
            self.logger.info(f"Processing {len(items)} items from page {page_count}")
            
            for item in items:
                yield item
            
            url = data.get('metadata', {}).get('nextPage')
            if url:
                await asyncio.sleep(0.3)

