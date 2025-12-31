"""Meilisearch hit relevance filtering for Mode 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .tag_text_matcher import TagTextMatcher


@dataclass(frozen=True)
class MeilisearchHitRelevanceFilter:
    """Decides whether a raw Meilisearch hit is relevant for a given tag.

    This is a **pre-download** filter intended to reduce bandwidth usage by
    checking the prompt (if available) or falling back to `tagNames`.
    """

    matcher: TagTextMatcher = TagTextMatcher()

    def is_relevant(self, hit: Dict[str, Any], tag: str) -> Tuple[bool, str]:
        if not isinstance(hit, dict):
            return False, "Invalid Meilisearch hit (not a dict)"

        # 1. Check prompt text (if available). This is the strongest signal.
        prompt = hit.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            if self.matcher.matches_text(prompt, tag):
                return True, "Mode 6 prefilter: prompt text matched"
            
            # If prompt is present but doesn't match, we reject it to avoid "spam tagged" images.
            return False, "Mode 6 prefilter: prompt text mismatch"

        # 2. Fallback: Check tagNames if prompt was missing.
        tag_names_raw = hit.get("tagNames")
        tag_names: List[str] = []

        if isinstance(tag_names_raw, list):
            tag_names = [t for t in tag_names_raw if isinstance(t, str) and t.strip()]
        elif isinstance(tag_names_raw, str) and tag_names_raw.strip():
            tag_names = [tag_names_raw.strip()]

        if not tag_names:
            return False, "Mode 6 prefilter: hit missing tagNames and prompt"

        if self.matcher.tag_list_contains_tag(tag_names, tag):
            return True, "Mode 6 prefilter: tagNames matched"

        return False, "Mode 6 prefilter: tagNames mismatch"
