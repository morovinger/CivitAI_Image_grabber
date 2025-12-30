"""Main runner that orchestrates the download process."""

import os
import sys
import asyncio
import logging
from typing import Optional, List, Dict, Any

from .config import (
    parse_arguments, setup_logging, SCRIPT_VERSION,
    RED, GREEN, YELLOW, RESET
)
from .utils import sanitize_tag_for_api, sanitize_tag_for_directory, clean_path_component
from .database import ImageTracker, NullTracker
from .api import CivitaiAPI
from .downloader import ImageDownloader
from .meilisearch_client import (
    CivitaiMeilisearchClient,
    CivitaiMeilisearchAuthError,
    CivitaiMeilisearchRateLimitError,
)
from .meilisearch_hit_transformer import MeilisearchHitTransformer


class CivitaiRunner:
    """Main runner class for CivitAI downloader."""
    
    def __init__(self, args=None):
        """Initialize the runner."""
        self.args = args or parse_arguments()
        self.logger = setup_logging()
        
        self._interactive = self.args.mode is None
        
        self.api: Optional[CivitaiAPI] = None
        self.tracker: Optional[ImageTracker] = None
        self.downloader: Optional[ImageDownloader] = None
        
        self.mode: Optional[str] = None
        self.quality: str = "SD"
        self.allow_redownload: bool = False
        self.disable_prompt_check: bool = False
        self.skip_db_tracking: bool = True  # Default: skip tracking
        
        self.run_results: Dict[str, Dict[str, Any]] = {}
    
    def _setup_config(self) -> bool:
        """Setup configuration from args or interactive input."""
        # Quality
        if self.args.quality:
            self.quality = "SD" if self.args.quality == 1 else "HD"
        elif self._interactive:
            choice = input("Select quality (1=SD, 2=HD) [1]: ").strip()
            self.quality = "HD" if choice == '2' else "SD"
        
        # Skip DB tracking
        if self.args.skip_db_tracking:
            self.skip_db_tracking = self.args.skip_db_tracking == 'y'
        elif self._interactive:
            choice = input("Skip database image tracking? (y/n) [y]: ").strip().lower()
            self.skip_db_tracking = choice != 'n'
        
        # Redownload (only relevant if tracking is enabled)
        if not self.skip_db_tracking:
            if self.args.redownload:
                self.allow_redownload = self.args.redownload == 1
            elif self._interactive:
                choice = input("Allow re-downloading tracked images? (1=Yes, 2=No) [2]: ").strip()
                self.allow_redownload = choice == '1'
        else:
            self.allow_redownload = True  # If tracking disabled, always allow download
        
        # Mode
        if self.args.mode:
            self.mode = str(self.args.mode)
        elif self._interactive:
            print("\nDownload modes:")
            print("  1 = By username")
            print("  2 = By model ID")
            print("  3 = By model tag (searches models with tag, downloads their gallery images)")
            print("  4 = By model version ID")
            print("  5 = By direct image tag (searches ALL images tagged via public API)")
            print("  6 = By website search (Meilisearch - no scrolling - finds 100k+ results)")
            choice = input("Choose mode: ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                self.mode = choice
            else:
                print("Invalid mode selection.")
                return False
        else:
            self.logger.error("Mode is required in non-interactive mode.")
            return False
        
        # Prompt check (for tag modes 3 and 5)
        if self.mode in ['3', '5']:
            if self.args.disable_prompt_check:
                self.disable_prompt_check = self.args.disable_prompt_check == 'y'
            elif self._interactive:
                print(f"\n{YELLOW}Prompt check filters images by checking if tag words are in the prompt.{RESET}")
                print("This helps ensure images are actually related to the tag.")
                choice = input("Disable prompt check? (y/n) [n]: ").strip().lower()
                self.disable_prompt_check = choice == 'y'
        
        return True
    
    def _get_identifiers(self) -> List[str]:
        """Get identifiers based on mode."""
        if self.mode == '1':
            if self.args.username:
                return [u.strip() for u in self.args.username.split(',') if u.strip()]
            elif self._interactive:
                inp = input("Enter username(s) (comma-separated): ").strip()
                return [u.strip() for u in inp.split(',') if u.strip()]
        
        elif self.mode == '2':
            if self.args.model_id:
                return [m.strip() for m in self.args.model_id.split(',') if m.strip().isdigit()]
            elif self._interactive:
                inp = input("Enter model ID(s) (comma-separated): ").strip()
                return [m.strip() for m in inp.split(',') if m.strip().isdigit()]
        
        elif self.mode == '3':
            if self.args.tags:
                return [t.strip() for t in self.args.tags.split(',') if t.strip()]
            elif self._interactive:
                inp = input("Enter tag(s) (comma-separated): ").strip()
                return [t.strip() for t in inp.split(',') if t.strip()]
        
        elif self.mode == '4':
            if self.args.model_version_id:
                return [m.strip() for m in self.args.model_version_id.split(',') if m.strip().isdigit()]
            elif self._interactive:
                inp = input("Enter model version ID(s) (comma-separated): ").strip()
                return [m.strip() for m in inp.split(',') if m.strip().isdigit()]
        
        elif self.mode == '5':
            if self.args.tags:
                return [t.strip() for t in self.args.tags.split(',') if t.strip()]
            elif self._interactive:
                inp = input("Enter tag(s) for direct image search (comma-separated): ").strip()
                return [t.strip() for t in inp.split(',') if t.strip()]
        
        elif self.mode == '6':
            if self.args.tags:
                return [t.strip() for t in self.args.tags.split(',') if t.strip()]
            elif self._interactive:
                inp = input("Enter search term(s) for website search (comma-separated): ").strip()
                return [t.strip() for t in inp.split(',') if t.strip()]
        
        return []
    
    async def run(self) -> bool:
        """Run the download process."""
        print(f"--- Civitai Downloader v{SCRIPT_VERSION} ---")
        
        if self._interactive:
            print("Running in interactive mode.")
        else:
            print("Running in command-line mode.")
            self.logger.info("Running in command-line mode.")
        
        if not self._setup_config():
            return False
        
        try:
            # Use NullTracker if tracking is disabled
            self.tracker = NullTracker() if self.skip_db_tracking else ImageTracker()
            self.api = CivitaiAPI(
                timeout=self.args.timeout,
                semaphore_limit=self.args.semaphore_limit,
                retries=self.args.retries
            )
            self.downloader = ImageDownloader(
                api=self.api,
                tracker=self.tracker,
                output_dir=self.args.output_dir,
                quality=self.quality,
                allow_redownload=self.allow_redownload,
                max_path_length=self.args.max_path,
                disable_sorting=self.args.no_sort,
                skip_metadata=self.args.no_meta
            )
        except Exception as e:
            self.logger.critical(f"Failed to initialize components: {e}")
            return False
        
        self._log_config()
        
        identifiers = self._get_identifiers()
        if not identifiers:
            self.logger.error("No valid identifiers provided.")
            print("Error: No valid identifiers provided.")
            return False
        
        self.logger.info(f"Processing {len(identifiers)} identifier(s)...")
        
        try:
            if self.mode == '1':
                await self._process_usernames(identifiers)
            elif self.mode == '2':
                await self._process_model_ids(identifiers)
            elif self.mode == '3':
                await self._process_tags(identifiers)
            elif self.mode == '4':
                await self._process_model_version_ids(identifiers)
            elif self.mode == '5':
                await self._process_direct_tag_search(identifiers)
            elif self.mode == '6':
                await self._process_website_scraper(identifiers)
            
            return True
            
        except Exception as e:
            self.logger.critical(f"Error during processing: {e}", exc_info=True)
            return False
        
        finally:
            await self._cleanup()
    
    def _log_config(self) -> None:
        """Log current configuration."""
        self.logger.info("--- Initializing Downloader ---")
        self.logger.info(f"Mode: {self.mode}, Quality: {self.quality}")
        self.logger.info(f"DB Tracking: {'Disabled' if self.skip_db_tracking else 'Enabled'}")
        if not self.skip_db_tracking:
            self.logger.info(f"Redownload: {'Yes' if self.allow_redownload else 'No'}")
        self.logger.info(f"Output Directory: {self.args.output_dir}")
        self.logger.info(f"Semaphore Limit: {self.args.semaphore_limit}, Timeout: {self.args.timeout}s")
        self.logger.info(f"Sorting: {'Disabled' if self.args.no_sort else 'Enabled'}")
        if self.mode == '3':
            self.logger.info(f"Prompt Check: {'Disabled' if self.disable_prompt_check else 'Enabled'}")
        self.logger.info("-------------------------------")
    
    async def _process_usernames(self, usernames: List[str]) -> None:
        """Process username-based downloads."""
        option_folder = os.path.join(self.args.output_dir, "Username_Search")
        os.makedirs(option_folder, exist_ok=True)
        
        for username in usernames:
            self.logger.info(f"Processing username: {username}")
            target_dir = os.path.join(option_folder, clean_path_component(username))
            os.makedirs(target_dir, exist_ok=True)
            
            count = 0
            async for item in self.api.iter_images_by_username(username):
                success, path, reason = await self.downloader.download_single_image(
                    item, target_dir, tag=None, check_prompt=False
                )
                if success:
                    count += 1
                elif reason:
                    self.downloader.record_skip(reason)
            
            self.logger.info(f"Downloaded {count} images for user '{username}'")
            
            if not self.args.no_sort:
                await self.downloader.sort_images_by_model(target_dir)
    
    async def _process_model_ids(self, model_ids: List[str]) -> None:
        """Process model ID-based downloads."""
        option_folder = os.path.join(self.args.output_dir, "Model_ID_Search")
        os.makedirs(option_folder, exist_ok=True)
        
        for model_id in model_ids:
            self.logger.info(f"Processing model ID: {model_id}")
            target_dir = os.path.join(option_folder, f"model_{model_id}")
            os.makedirs(target_dir, exist_ok=True)
            
            count = 0
            async for item in self.api.iter_images_by_model(int(model_id)):
                success, path, reason = await self.downloader.download_single_image(
                    item, target_dir, tag=None, check_prompt=False
                )
                if success:
                    count += 1
                elif reason:
                    self.downloader.record_skip(reason)
            
            self.logger.info(f"Downloaded {count} images for model {model_id}")
            
            if not self.args.no_sort:
                await self.downloader.sort_images_by_model(target_dir)
    
    async def _process_model_version_ids(self, version_ids: List[str]) -> None:
        """Process model version ID-based downloads."""
        option_folder = os.path.join(self.args.output_dir, "Model_Version_ID_Search")
        os.makedirs(option_folder, exist_ok=True)
        
        for version_id in version_ids:
            self.logger.info(f"Processing model version ID: {version_id}")
            target_dir = os.path.join(option_folder, f"modelVersion_{version_id}")
            os.makedirs(target_dir, exist_ok=True)
            
            count = 0
            async for item in self.api.iter_images_by_model_version(int(version_id)):
                success, path, reason = await self.downloader.download_single_image(
                    item, target_dir, tag=None, check_prompt=False
                )
                if success:
                    count += 1
                elif reason:
                    self.downloader.record_skip(reason)
            
            self.logger.info(f"Downloaded {count} images for model version {version_id}")
            
            if not self.args.no_sort:
                await self.downloader.sort_images_by_model(target_dir)
    
    async def _process_tags(self, tags: List[str]) -> None:
        """Process tag-based downloads using model search."""
        option_folder = os.path.join(self.args.output_dir, "Model_Tag_Search")
        os.makedirs(option_folder, exist_ok=True)
        
        self.logger.info("--- Starting Tag Search Mode ---")
        self.logger.info(f"Prompt check: {'Disabled' if self.disable_prompt_check else 'Enabled'}")
        
        for tag in tags:
            self.logger.info(f"Processing tag: {tag}")
            tag_query = sanitize_tag_for_api(tag)  # Use space for API
            tag_dir_name = sanitize_tag_for_directory(tag)  # Use underscore for directory
            tag_dir = os.path.join(option_folder, tag_dir_name)
            os.makedirs(tag_dir, exist_ok=True)
            
            # Search for models with this tag
            print(f"\nSearching for models tagged with '{tag_query}'...")
            models = await self.api.search_models_by_tag(tag_query)
            
            if not models:
                self.logger.warning(f"No models found for tag '{tag}'")
                print(f"  No models found for tag '{tag}'")
                continue
            
            print(f"  Found {len(models)} models:")
            for mid, mname in models[:5]:
                print(f"    - {mname} (ID: {mid})")
            if len(models) > 5:
                print(f"    ... and {len(models) - 5} more")
            
            total_downloaded = 0
            total_skipped = 0
            
            # Process each model
            for model_id, model_name in models:
                model_dir = os.path.join(tag_dir, f"model_{model_id}")
                os.makedirs(model_dir, exist_ok=True)
                
                count = 0
                skipped = 0
                
                async for item in self.api.iter_images_by_model(model_id):
                    success, path, reason = await self.downloader.download_single_image(
                        item, model_dir, tag=tag, check_prompt=not self.disable_prompt_check
                    )
                    if success:
                        count += 1
                        total_downloaded += 1
                        if total_downloaded % 50 == 0:
                            print(f"  Downloaded {total_downloaded} images so far...")
                    elif reason:
                        self.downloader.record_skip(reason)
                        skipped += 1
                        total_skipped += 1
                
                self.logger.debug(f"Model {model_id}: {count} downloaded, {skipped} skipped")
                
                if not self.args.no_sort:
                    await self.downloader.sort_images_by_model(model_dir)
            
            print(f"\n  Tag '{tag}' complete: {total_downloaded} downloaded, {total_skipped} skipped")
            self.logger.info(f"Tag '{tag}': {total_downloaded} downloaded, {total_skipped} skipped")
        
        self.logger.info("--- Finished Tag Search Mode ---")
    
    async def _process_direct_tag_search(self, tags: List[str]) -> None:
        """Process direct image tag search - queries images API directly with tag parameter.
        
        This mode searches for images that are directly tagged, not just images from
        models tagged with the term. Results in significantly more images.
        """
        option_folder = os.path.join(self.args.output_dir, "Direct_Tag_Search")
        os.makedirs(option_folder, exist_ok=True)
        
        self.logger.info("--- Starting Direct Image Tag Search Mode ---")
        self.logger.info(f"Prompt check: {'Disabled' if self.disable_prompt_check else 'Enabled'}")
        
        for tag in tags:
            self.logger.info(f"Processing direct tag search: {tag}")
            tag_query = sanitize_tag_for_api(tag)
            tag_dir_name = sanitize_tag_for_directory(tag)
            tag_dir = os.path.join(option_folder, tag_dir_name)
            os.makedirs(tag_dir, exist_ok=True)
            
            print(f"\nSearching for images tagged with '{tag_query}'...")
            print("  (This searches images directly, not via models - may find 100k+ images)")
            
            total_downloaded = 0
            total_skipped = 0
            
            async for item in self.api.iter_images_by_tag(tag_query):
                success, path, reason = await self.downloader.download_single_image(
                    item, tag_dir, tag=tag, check_prompt=not self.disable_prompt_check
                )
                if success:
                    total_downloaded += 1
                    if total_downloaded % 100 == 0:
                        print(f"  Downloaded {total_downloaded} images so far...")
                elif reason:
                    self.downloader.record_skip(reason)
                    total_skipped += 1
            
            print(f"\n  Tag '{tag}' complete: {total_downloaded} downloaded, {total_skipped} skipped")
            self.logger.info(f"Direct tag search '{tag}': {total_downloaded} downloaded, {total_skipped} skipped")
            
            if not self.args.no_sort:
                await self.downloader.sort_images_by_model(tag_dir)
        
        self.logger.info("--- Finished Direct Image Tag Search Mode ---")
    
    async def _process_website_scraper(self, tags: List[str]) -> None:
        """Process website search mode using Meilisearch (no scrolling)."""
        option_folder = os.path.join(self.args.output_dir, "Website_Scraper_Search")
        os.makedirs(option_folder, exist_ok=True)
        
        self.logger.info("--- Starting Website Search Mode ---")

        try:
            meili = CivitaiMeilisearchClient(timeout_s=self.args.timeout)
        except CivitaiMeilisearchAuthError as e:
            # Fail fast: this mode cannot work without the token.
            self.logger.error(str(e))
            print(f"\n{RED}ERROR:{RESET} {e}")
            return
        transformer = MeilisearchHitTransformer()
        
        for tag in tags:
            self.logger.info(f"Processing website search: {tag}")
            tag_dir_name = sanitize_tag_for_directory(tag)
            tag_dir = os.path.join(option_folder, tag_dir_name)
            os.makedirs(tag_dir, exist_ok=True)
            
            print(f"\nStarting Meilisearch website-like search for '{tag}'...")
            print("  (No browser scrolling; results are paginated via offset/limit)")
            
            total_downloaded = 0
            total_skipped = 0
            seen_ids: set[str] = set()
            
            try:
                async for hit in meili.iter_search_hits(tag, filters=["poi != true"]):
                    item = transformer.transform(hit)
                    if not item:
                        continue

                    image_id = item.get("id")
                    if image_id is None:
                        continue
                    image_id_str = str(image_id)
                    if image_id_str in seen_ids:
                        continue
                    seen_ids.add(image_id_str)

                    success, path, reason = await self.downloader.download_single_image(
                        item, tag_dir, tag=tag, check_prompt=False # Website results are already filtered
                    )
                    
                    if success:
                        total_downloaded += 1
                        if total_downloaded % 50 == 0:
                            print(f"  Downloaded {total_downloaded} images so far...")
                    elif reason:
                        self.downloader.record_skip(reason)
                        total_skipped += 1
                        
            except CivitaiMeilisearchRateLimitError as e:
                self.logger.error(str(e))
                print(f"\n{YELLOW}WARNING:{RESET} {e}")
            except Exception as e:
                self.logger.error(f"Website search error: {e}")
                print(f"  Website search encountered an error: {e}")
            
            print(f"\n  Search '{tag}' complete: {total_downloaded} downloaded, {total_skipped} skipped")
            self.logger.info(f"Website search '{tag}': {total_downloaded} downloaded, {total_skipped} skipped")
            
            if not self.args.no_sort:
                await self.downloader.sort_images_by_model(tag_dir)
        
        self.logger.info("--- Finished Website Search Mode ---")

        await meili.close()

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Run finalization steps...")
        
        if self.downloader:
            print(self.downloader.get_stats_summary())
            self.logger.info(f"Stats: Downloaded={self.downloader.stats['downloaded']}, Skipped={self.downloader.stats['skipped']}")
        
        if self.api:
            await self.api.close()
        
        if self.tracker:
            self.tracker.close()
        
        print("\nDownload process complete.")


def main():
    """Main entry point."""
    runner = CivitaiRunner()
    
    try:
        success = asyncio.run(runner.run())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logging.getLogger('CivitaiDownloader').critical(f"Unhandled error: {e}", exc_info=True)
        print(f"\nCritical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

