"""Utility functions for CivitAI Downloader."""

import re
import os
from typing import Dict, Any, Optional


def detect_extension(data: bytes) -> Optional[str]:
    """Detects file extension from magic numbers (file header bytes)."""
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return ".png"
    if data.startswith(b'\xff\xd8\xff'):
        return ".jpeg"
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return ".webp"
    if len(data) >= 8 and data[4:8] == b'ftyp':
        return ".mp4"
    if data.startswith(b'\x1A\x45\xDF\xA3'):
        return ".webm"
    return None


def extract_image_meta(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts the actual metadata dict from an API item, handling nested structure.
    
    CivitAI API changed structure from:
        item["meta"] = {"prompt": "...", "Model": "..."}
    To:
        item["meta"] = {"id": 123, "meta": {"prompt": "...", "Model": "..."}}
    
    This function handles both old and new structures.
    """
    meta_field = item.get("meta")
    if not meta_field or not isinstance(meta_field, dict):
        return {}
    
    # Check for new nested structure: meta.meta exists and contains generation params
    nested_meta = meta_field.get("meta")
    if nested_meta and isinstance(nested_meta, dict):
        # New structure: actual metadata is in meta.meta
        return nested_meta
    
    # Old structure or meta doesn't have nested meta: check if prompt/Model exists at top level
    if "prompt" in meta_field or "Model" in meta_field or "seed" in meta_field:
        return meta_field
    
    # Fallback: return empty dict if no recognizable structure
    return {}


def clean_path_component(name: str, max_length: int = 100) -> str:
    """Cleans a string for use as a file/directory name component."""
    if not name:
        return "unknown"
    
    # Remove or replace problematic characters
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name))
    cleaned = re.sub(r'[\s_]+', '_', cleaned).strip('_. ')
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip('_. ')
    
    return cleaned if cleaned else "unknown"


def sanitize_tag_for_api(tag: str) -> str:
    """Converts tag format for API query (underscore to space)."""
    return tag.replace("_", " ")


def sanitize_tag_for_directory(tag: str) -> str:
    """Converts tag format for directory name (space to underscore)."""
    return tag.replace(" ", "_")


def check_prompt_contains_tag(prompt: str, tag: str) -> bool:
    """Checks if a tag is present in the prompt (with flexible matching).
    
    Args:
        prompt: The image prompt text
        tag: The tag to check (can have underscores or spaces)
    
    Returns:
        True if the tag (or its variations) is found in the prompt
    """
    if not prompt:
        return False
    
    prompt_lower = prompt.lower()
    tag_lower = tag.lower()
    
    # Create variations of the tag to check
    # e.g., "star butterfly" -> ["star butterfly", "star_butterfly", "starbutterfly"]
    variations = [
        tag_lower,
        tag_lower.replace(" ", "_"),
        tag_lower.replace(" ", ""),
        tag_lower.replace("_", " "),
        tag_lower.replace("_", ""),
    ]
    
    # Check if any variation exists in the prompt
    for variation in variations:
        if variation in prompt_lower:
            return True
    
    # Fallback: check if ALL words are present (for partial matches)
    tag_words = [w for w in tag_lower.replace("_", " ").split() if w]
    return all(word in prompt_lower for word in tag_words)


def get_quality_suffix(quality: str) -> str:
    """Returns the URL suffix for quality selection."""
    return "" if quality == "HD" else ".width=512"


def truncate_path_if_needed(base_path: str, filename: str, max_total: int = 240) -> str:
    """Truncates filename if full path would exceed max length."""
    full_path = os.path.join(base_path, filename)
    if len(full_path) <= max_total:
        return filename
    
    # Calculate how much to truncate
    name, ext = os.path.splitext(filename)
    available = max_total - len(base_path) - len(ext) - 1  # -1 for path separator
    
    if available > 10:  # Keep at least 10 chars
        return name[:available] + ext
    
    return filename  # Can't truncate safely, return as-is

