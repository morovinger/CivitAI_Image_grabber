"""Website scraper for CivitAI using Playwright."""

import logging
import asyncio
import json
import random
from typing import AsyncGenerator, Dict, Any, Optional, List
from urllib.parse import quote

from playwright.async_api import async_playwright, Page, Response

from .config import DEFAULT_TIMEOUT


class CivitaiScraper:
    """Handles website scraping and traffic interception."""
    
    def __init__(self, headless: bool = True):
        self.logger = logging.getLogger('CivitaiDownloader')
        self.headless = headless
        self.page: Optional[Page] = None
        self._items_queue: asyncio.Queue = asyncio.Queue()
        self._processing_done = asyncio.Event()
        self._found_count = 0
        
    async def search_and_intercept(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Navigates to search page and yields image items intercepted from internal API.
        
        Args:
            query: Search tag/term
            
        Yields:
            Image dictionaries similar to API structure
        """
        encoded_query = quote(query)
        # Search URL sorting by 'Most Reactions' (images_v6 index) to match popular results
        search_url = f"https://civitai.com/search/images?sortBy=images_v6&query={encoded_query}"
        
        self.logger.info(f"Starting browser (headless={self.headless})...")
        
        async with async_playwright() as p:
            # Launch browser with stealthier args
            browser = await p.chromium.launch(
                headless=self.headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-setuid-sandbox'
                ]
            )
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            page = await context.new_page()
            self.page = page
            
            # Setup interception
            page.on("response", self._handle_response)
            
            self.logger.info(f"Navigating to: {search_url}")
            try:
                await page.goto(search_url, timeout=DEFAULT_TIMEOUT * 1000)
                
                # Wait for initial load
                self.logger.info("Waiting for initial results...")
                await page.wait_for_timeout(5000)
                
                # Scroll loop
                no_new_results_count = 0
                last_count = 0
                
                while True:
                    # Scroll down to trigger loading
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(random.randint(1500, 3000))
                    
                    # Yield any items found so far
                    while not self._items_queue.empty():
                        item = await self._items_queue.get()
                        yield item
                    
                    # Check if we're still finding things
                    if self._found_count == last_count:
                        no_new_results_count += 1
                        self.logger.debug(f"No new items found (attempt {no_new_results_count}/5)")
                        
                        # Try to find "end of results" or load more button
                        if no_new_results_count >= 5:
                            self.logger.info("No new items found after multiple scrolls. Stopping.")
                            break
                    else:
                        no_new_results_count = 0
                        self.logger.info(f"Scraped {self._found_count} images so far...")
                        last_count = self._found_count
                    
            except Exception as e:
                self.logger.error(f"Browser automation error: {e}")
            finally:
                await browser.close()
                self.logger.info("Browser closed.")

    async def _handle_response(self,response: Response):
        """Intercepts and parses network responses."""
        try:
            url = response.url
            
            # Check for Meilisearch / multi-search endpoint (the main one used by search page)
            if "multi-search" in url and response.status == 200 and response.request.method == "POST":
                try:
                    data = await response.json()
                    await self._parse_meilisearch_response(data)
                except Exception as e:
                    pass # Ignore JSON parse errors on non-JSON responses
            
            # Check for TRPC calls (infinite scroll often uses this)
            elif "image.getInfinite" in url and response.status == 200:
                try:
                    data = await response.json()
                    await self._parse_trpc_response(data)
                except Exception:
                    pass

        except Exception as e:
            self.logger.debug(f"Error handling response: {e}")

    async def _parse_meilisearch_response(self, data: Dict[str, Any]):
        """Extracts images from Meilisearch response."""
        results = data.get('results', [])
        for result in results:
            hits = result.get('hits', [])
            for hit in hits:
                # Transform Meilisearch hit to standard API structure
                item = self._transform_hit(hit)
                if item:
                    await self._items_queue.put(item)
                    self._found_count += 1

    async def _parse_trpc_response(self, data: Dict[str, Any]):
        """Extracts images from TRPC response."""
        # TRPC structure can be complex and nested
        try:
            # Handle array response
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            
            result = data.get('result', {}).get('data', {})
            
            # Items might be directly in data or nested
            items = []
            if 'items' in result:
                items = result['items']
            elif isinstance(result, list):
                items = result
                
            for hit in items:
                item = self._transform_hit(hit)
                if item:
                    await self._items_queue.put(item)
                    self._found_count += 1
        except Exception:
            pass

    def _transform_hit(self, hit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalizes different response formats into a standard structure."""
        if not hit or 'id' not in hit:
            return None
            
        # Extract basic fields
        image_id = hit.get('id')
        
        # URL construction - Meilisearch results often have just the hash or partial URL
        url = hit.get('url')
        if not url and 'hash' in hit:
             # Fallback if URL is missing but hash exists (unlikely but possible)
             pass
        
        if not url:
             return None
             
        # Ensure full URL
        if not url.startswith('http'):
            # CivitAI usually uses image.civitai.com
            # We construct the URL with original=true to ensure high quality downloads
            url = f"https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/{url}/original=true,quality=90/{image_id}.jpeg"

        # Create standard structure
        item = {
            'id': image_id,
            'url': url,
            'nsfw': hit.get('nsfw', False),
            'createdAt': hit.get('createdAt'),
            'meta': hit.get('meta', {})
        }
        
        # If meta prompt is missing but 'prompt' is at top level (common in search results)
        if 'prompt' not in item['meta'] and 'prompt' in hit:
             if item['meta'] is None: item['meta'] = {}
             item['meta']['prompt'] = hit['prompt']
             
        return item

