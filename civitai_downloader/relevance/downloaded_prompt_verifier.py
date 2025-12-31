"""Post-download prompt verification for Mode 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..metadata.pillow_prompt_extractor import PillowPromptExtractor
from .tag_text_matcher import TagTextMatcher


@dataclass(frozen=True)
class DownloadedPromptVerifier:
    """Verifies a downloaded asset contains the tag in embedded prompt metadata."""

    extractor: PillowPromptExtractor = PillowPromptExtractor()
    matcher: TagTextMatcher = TagTextMatcher()

    def verify(self, image_bytes: bytes, tag: str) -> Tuple[bool, Optional[str], str]:
        prompt_text, source_reason = self.extractor.extract_prompt(image_bytes)
        if not prompt_text:
            return False, None, f"Mode 6 verify: no embedded prompt ({source_reason})"

        if self.matcher.matches_text(prompt_text, tag):
            return True, prompt_text, "Mode 6 verify: embedded prompt matched"

        return False, prompt_text, "Mode 6 verify: embedded prompt mismatch"



