"""Tag text matching utilities.

This module centralizes tag matching logic so we can reuse it for:
- prompt text verification (substring / word-based checks)
- Meilisearch hit tag list verification (exact-ish matching after normalization)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Set


_WHITESPACE_RE = re.compile(r"\s+")
_TAG_SEPARATORS_RE = re.compile(r"[_\-]+")


@dataclass(frozen=True)
class TagTextMatcher:
    """Matches a user-provided tag against prompt text and tag lists."""

    def normalize_tag(self, tag: str) -> str:
        """Normalize a tag for comparisons (lowercase, separators -> spaces)."""
        raw = str(tag or "").strip().lower()
        raw = _TAG_SEPARATORS_RE.sub(" ", raw)
        raw = _WHITESPACE_RE.sub(" ", raw).strip()
        return raw

    def build_variations(self, tag: str) -> Set[str]:
        """Build common variations for substring matching."""
        canonical = self.normalize_tag(tag)
        # e.g. "star butterfly" -> {"star butterfly", "star_butterfly", "starbutterfly"}
        return {
            canonical,
            canonical.replace(" ", "_"),
            canonical.replace(" ", ""),
        }

    def matches_text(self, text: str, tag: str) -> bool:
        """Check if tag is present in a blob of text (prompt/parameters)."""
        if not text:
            return False

        text_lower = str(text).lower()
        variations = self.build_variations(tag)

        for variation in variations:
            if variation and variation in text_lower:
                return True

        # Fallback: require every word to appear somewhere in the text.
        canonical = self.normalize_tag(tag)
        words = [w for w in canonical.split(" ") if w]
        return bool(words) and all(word in text_lower for word in words)

    def tag_list_contains_tag(self, tag_names: Iterable[str], tag: str) -> bool:
        """Check if a list of tag names contains the requested tag."""
        canonical = self.normalize_tag(tag)
        if not canonical:
            return False

        wanted = {
            canonical,
            canonical.replace(" ", "_"),
            canonical.replace(" ", ""),
        }

        for t in tag_names or []:
            if not isinstance(t, str):
                continue
            tn = self.normalize_tag(t)
            if not tn:
                continue

            candidates = {
                tn,
                tn.replace(" ", "_"),
                tn.replace(" ", ""),
            }

            if wanted.intersection(candidates):
                return True

        return False



