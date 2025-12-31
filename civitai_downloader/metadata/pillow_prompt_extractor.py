"""Embedded prompt extraction using Pillow.

We use this for Mode 6 post-download verification: extract prompt/parameters text
from PNG/JPEG/WebP metadata and verify it contains the requested tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class PillowPromptExtractor:
    """Extracts prompt-ish text from common image metadata locations."""

    # Common keys used by generators / upload pipelines
    INFO_KEYS = (
        "parameters",  # Automatic1111 / ComfyUI exports (common)
        "prompt",
        "Prompt",
        "comment",
        "Comment",
        "description",
        "Description",
    )

    def extract_prompt(self, image_bytes: bytes) -> Tuple[Optional[str], str]:
        """Return (prompt_text, source_reason). prompt_text is None if not found."""
        if not image_bytes:
            return None, "No bytes provided"

        try:
            from PIL import Image
        except Exception as e:  # pragma: no cover
            return None, f"Pillow is not installed: {e}"

        try:
            with Image.open(BytesIO(image_bytes)) as img:
                # Some formats lazily load; this forces metadata parsing.
                img.load()

                # 1) PNG text / generic info dict
                info: Dict[str, Any] = getattr(img, "info", {}) or {}
                for key in self.INFO_KEYS:
                    val = info.get(key)
                    text = self._decode_text_value(val)
                    if text:
                        return text, f"Found in img.info['{key}']"

                # Pillow sometimes exposes PNG text chunks via `.text` (dict-like).
                text_dict = getattr(img, "text", None)
                if isinstance(text_dict, dict):
                    for key in self.INFO_KEYS:
                        val = text_dict.get(key)
                        text = self._decode_text_value(val)
                        if text:
                            return text, f"Found in img.text['{key}']"

                # 2) EXIF tags (JPEG/WebP/PNG-with-exif)
                exif = None
                try:
                    exif = img.getexif()
                except Exception:
                    exif = None

                if exif:
                    # Prefer tags most likely to contain parameters.
                    # - 37510: UserComment
                    # - 270: ImageDescription
                    # - 40092: XPComment (UTF-16LE)
                    for tag_id, label in (
                        (37510, "UserComment"),
                        (270, "ImageDescription"),
                        (40092, "XPComment"),
                    ):
                        val = exif.get(tag_id)
                        text = self._decode_exif_value(tag_id, val)
                        if text:
                            return text, f"Found in EXIF {label} ({tag_id})"

        except Exception as e:
            # Unsupported formats (e.g., mp4) or corrupt bytes.
            return None, f"Pillow failed to open/parse: {type(e).__name__}: {e}"

        return None, "No embedded prompt found"

    @staticmethod
    def _decode_text_value(val: Any) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, str):
            txt = val.strip()
            return txt or None
        if isinstance(val, bytes):
            for enc in ("utf-8", "utf-16le", "latin-1"):
                try:
                    txt = val.decode(enc, errors="ignore").strip()
                    if txt:
                        return txt
                except Exception:
                    continue
        return None

    def _decode_exif_value(self, tag_id: int, val: Any) -> Optional[str]:
        if val is None:
            return None

        # Pillow may return bytes for some EXIF values
        if tag_id == 37510:  # UserComment
            return self._decode_exif_user_comment(val)

        # XPComment is UTF-16LE bytes
        if tag_id == 40092 and isinstance(val, (bytes, bytearray)):
            try:
                txt = bytes(val).decode("utf-16le", errors="ignore").strip("\x00").strip()
                return txt or None
            except Exception:
                return None

        return self._decode_text_value(val)

    @staticmethod
    def _decode_exif_user_comment(val: Any) -> Optional[str]:
        # EXIF UserComment often starts with an encoding prefix:
        # b"ASCII\0\0\0", b"UNICODE\0", b"JIS\0\0\0\0\0"
        if isinstance(val, str):
            txt = val.strip()
            return txt or None

        if not isinstance(val, (bytes, bytearray)):
            return None

        data = bytes(val)
        if len(data) < 8:
            return None

        prefix = data[:8]
        payload = data[8:]

        if prefix.startswith(b"ASCII"):
            return payload.decode("utf-8", errors="ignore").strip("\x00").strip() or None
        if prefix.startswith(b"UNICODE"):
            return payload.decode("utf-16", errors="ignore").strip("\x00").strip() or None
        if prefix.startswith(b"JIS"):
            return payload.decode("shift_jis", errors="ignore").strip("\x00").strip() or None

        # Unknown prefix: best-effort decode whole blob
        for enc in ("utf-8", "utf-16le", "latin-1"):
            try:
                txt = data.decode(enc, errors="ignore").strip("\x00").strip()
                if txt:
                    return txt
            except Exception:
                continue

        return None



