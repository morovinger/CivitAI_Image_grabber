"""Core image downloading functionality."""

import os
import asyncio
import aiofiles
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from .config import DEFAULT_MAX_PATH_LENGTH
from .utils import (
    detect_extension, extract_image_meta, clean_path_component,
    check_prompt_contains_tag, get_quality_suffix, truncate_path_if_needed
)
from .database import ImageTracker
from .api import CivitaiAPI
from .metadata.pillow_prompt_extractor import PillowPromptExtractor
from .relevance.tag_text_matcher import TagTextMatcher


class ImageDownloader:
    """Handles downloading and saving images from CivitAI."""
    
    def __init__(
        self,
        api: CivitaiAPI,
        tracker: ImageTracker,
        output_dir: str,
        quality: str = "HD",
        allow_redownload: bool = False,
        max_path_length: int = DEFAULT_MAX_PATH_LENGTH,
        disable_sorting: bool = False,
        skip_metadata: bool = False
    ):
        """Initialize the downloader.
        
        Args:
            api: CivitAI API client instance
            tracker: Image tracking database instance
            output_dir: Base output directory
            quality: "SD" or "HD"
            allow_redownload: Whether to re-download tracked images
            max_path_length: Maximum path length
            disable_sorting: Disable sorting into model subfolders
            skip_metadata: Skip creating _meta.txt files
        """
        self.logger = logging.getLogger('CivitaiDownloader')
        self.api = api
        self.tracker = tracker
        self.output_dir = os.path.abspath(output_dir)
        self.quality = quality
        self.allow_redownload = allow_redownload
        self.max_path_length = max_path_length
        self.disable_sorting = disable_sorting
        self.skip_metadata = skip_metadata
        
        # Statistics
        self.stats = {
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'no_meta': 0
        }
        self.skip_reasons: Dict[str, int] = {}
    
    def _get_image_url(self, item: Dict[str, Any]) -> Optional[str]:
        """Get the appropriate image URL based on quality setting."""
        url = item.get('url')
        if not url:
            return None
        
        if self.quality == "SD":
            return f"{url}.width=512" if not url.endswith('.width=512') else url
        return url
    
    async def download_single_image(
        self,
        item: Dict[str, Any],
        target_dir: str,
        tag: Optional[str] = None,
        check_prompt: bool = True
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Download a single image.
        
        Args:
            item: Image item from API
            target_dir: Directory to save to
            tag: Optional tag for prompt checking and tracking
            check_prompt: Whether to verify tag in prompt
        
        Returns:
            Tuple of (success, file_path, skip_reason)
        """
        image_id = item.get('id')
        if not image_id:
            return False, None, "No image ID"
        
        image_id_str = str(image_id)
        
        # Check if already downloaded
        if not self.allow_redownload and self.tracker.is_image_tracked(image_id_str, self.quality):
            return False, None, "Already tracked"
        
        # Extract metadata
        meta = extract_image_meta(item)
        prompt = meta.get('prompt', '')
        
        # Check prompt contains tag if required (Pre-download check)
        prompt_verified_via_api = False
        if check_prompt and tag:
            matcher = TagTextMatcher()
            if prompt and str(prompt).strip():
                # If API has prompt, it MUST match.
                if matcher.matches_text(prompt, tag):
                    prompt_verified_via_api = True
                else:
                    return False, None, f"Prompt check failed (API): {tag}"
            else:
                 # API prompt missing, proceed to download and check later
                 prompt_verified_via_api = False
        
        # Get image URL
        url = self._get_image_url(item)
        if not url:
            return False, None, "No URL"
        
        # Download image bytes
        image_bytes = await self.api.fetch_image_bytes(url)
        if not image_bytes:
            return False, None, "Download failed"

        # Post-download verification (Hybrid Mode)
        # If we wanted to check prompt, but haven't verified it via API yet (because it was missing),
        # we check the file now.
        if check_prompt and tag and not prompt_verified_via_api:
            extractor = PillowPromptExtractor()
            extracted_prompt, source = extractor.extract_prompt(image_bytes)
            
            if not extracted_prompt:
                # If we couldn't verify via API AND couldn't find prompt in file -> Reject
                return False, None, "Prompt check failed: No prompt found in API or Image"
            
            matcher = TagTextMatcher()
            if not matcher.matches_text(extracted_prompt, tag):
                 return False, None, f"Prompt check failed (Image Metadata): {tag}"
            
            # If we got here, it matched!
        
        # Detect file extension
        ext = detect_extension(image_bytes)
        if not ext:
            # Try to get from URL
            url_lower = url.lower()
            if '.png' in url_lower:
                ext = '.png'
            elif '.jpg' in url_lower or '.jpeg' in url_lower:
                ext = '.jpeg'
            elif '.webp' in url_lower:
                ext = '.webp'
            else:
                ext = '.jpeg'  # Default
        
        # Create filename
        filename = f"{image_id_str}{ext}"
        filename = truncate_path_if_needed(target_dir, filename, self.max_path_length)
        file_path = os.path.join(target_dir, filename)
        
        # Ensure directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Save image
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(image_bytes)
        except Exception as e:
            self.logger.error(f"Failed to save image {image_id}: {e}")
            return False, None, f"Save failed: {e}"
        
        # Save metadata (unless disabled)
        if not self.skip_metadata:
            await self._save_metadata(item, target_dir, image_id_str)
        
        # Track in database
        checkpoint_name = meta.get('Model', '') if meta else None
        tags_list = [tag] if tag else []
        self.tracker.mark_image_downloaded(
            image_id_str, file_path, self.quality,
            tags=tags_list, url=item.get('url'), checkpoint_name=checkpoint_name
        )
        
        # Update stats
        self.stats['downloaded'] += 1
        if not meta or not any(meta.values()):
            self.stats['no_meta'] += 1
        
        return True, file_path, None
    
    async def _save_metadata(
        self,
        item: Dict[str, Any],
        target_dir: str,
        image_id: str
    ) -> None:
        """Save image metadata to a text file."""
        meta = extract_image_meta(item)
        if not meta:
            return
        
        meta_path = os.path.join(target_dir, f"{image_id}_meta.txt")
        
        try:
            lines = []
            for key, value in meta.items():
                if value is not None and str(value).strip():
                    lines.append(f"{key}: {value}")
            
            # Add username if available
            username = item.get('username')
            if username:
                lines.insert(0, f"Username: {username}")
            
            if lines:
                async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
                    await f.write('\n'.join(lines))
        except Exception as e:
            self.logger.debug(f"Failed to save metadata for {image_id}: {e}")
    
    def record_skip(self, reason: str) -> None:
        """Record a skip reason in statistics."""
        self.stats['skipped'] += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
    
    def record_failure(self, reason: str) -> None:
        """Record a failure in statistics."""
        self.stats['failed'] += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
    
    def get_stats_summary(self) -> str:
        """Get a formatted statistics summary."""
        lines = [
            "\n--- Download Statistics Summary ---",
            f"Total successful downloads: {self.stats['downloaded']}",
            f"Total skipped: {self.stats['skipped']}",
            f"Total failed: {self.stats['failed']}",
            f"Images without metadata: {self.stats['no_meta']}",
        ]
        
        if self.skip_reasons:
            lines.append("\nReasons for skipping/failing:")
            for reason, count in sorted(self.skip_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  - {reason}: {count}")
        
        lines.append("-" * 35)
        return '\n'.join(lines)
    
    async def sort_images_by_model(self, directory: str) -> None:
        """Sort downloaded images into subfolders by checkpoint/model name."""
        if self.disable_sorting:
            return
        
        self.logger.info(f"Starting sort process in: {directory}")
        
        if not os.path.isdir(directory):
            return
        
        meta_files = [f for f in os.listdir(directory) if f.endswith('_meta.txt')]
        
        if not meta_files:
            self.logger.info(f"No metadata files found to process in {directory}")
            return
        
        sorted_count = 0
        invalid_dir = os.path.join(directory, "invalid_metadata")
        no_meta_dir = os.path.join(directory, "no_metadata")
        
        for meta_file in meta_files:
            image_id = meta_file.replace('_meta.txt', '')
            meta_path = os.path.join(directory, meta_file)
            
            # Find corresponding image file
            image_file = None
            for ext in ['.png', '.jpeg', '.jpg', '.webp']:
                candidate = os.path.join(directory, f"{image_id}{ext}")
                if os.path.exists(candidate):
                    image_file = candidate
                    break
            
            if not image_file:
                continue
            
            # Read metadata to get model name
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                model_name = None
                for line in content.split('\n'):
                    if line.startswith('Model:'):
                        model_name = line.split(':', 1)[1].strip()
                        break
                
                if model_name:
                    # Create model subdirectory
                    model_dir = os.path.join(directory, clean_path_component(model_name))
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Move files
                    import shutil
                    shutil.move(image_file, os.path.join(model_dir, os.path.basename(image_file)))
                    shutil.move(meta_path, os.path.join(model_dir, meta_file))
                    sorted_count += 1
                else:
                    # No model name - move to invalid_metadata
                    os.makedirs(invalid_dir, exist_ok=True)
                    import shutil
                    shutil.move(image_file, os.path.join(invalid_dir, os.path.basename(image_file)))
                    shutil.move(meta_path, os.path.join(invalid_dir, meta_file))
                    
            except Exception as e:
                self.logger.debug(f"Error sorting {image_id}: {e}")
        
        if sorted_count > 0:
            self.logger.info(f"Sorted {sorted_count} images in {directory}")
