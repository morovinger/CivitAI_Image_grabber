"""Transforms CivitAI Meilisearch hits into downloader-compatible items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MeilisearchHitTransformer:
    """Normalizes Meilisearch hits to the structure expected by ImageDownloader."""

    image_cdn_prefix: str = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA"

    def transform(self, hit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(hit, dict):
            return None

        hit_id = hit.get("id")
        if hit_id is None:
            return None

        # In Meilisearch hits, `url` is typically the asset UUID (not a full URL).
        asset_id = hit.get("url")
        if not isinstance(asset_id, str) or not asset_id.strip():
            return None
        asset_id = asset_id.strip()

        mime_type = hit.get("mimeType")
        ext = self._extension_from_mime_type(mime_type) or self._default_extension_for_type(hit.get("type"))

        full_url = asset_id
        if not full_url.startswith("http"):
            # Verified working pattern:
            # https://image.civitai.com/<cdnScope>/<asset_uuid>/original=true/<asset_uuid>.<ext>
            full_url = f"{self.image_cdn_prefix}/{asset_id}/original=true/{asset_id}{ext}"

        username = None
        user_obj = hit.get("user")
        if isinstance(user_obj, dict):
            username = user_obj.get("username")
        if not isinstance(username, str):
            username = hit.get("username")

        nsfw_level = hit.get("nsfwLevel")
        nsfw = bool(isinstance(nsfw_level, int) and nsfw_level > 1)

        # Meilisearch tends to expose minimal metadata. Preserve what we have
        # and always carry prompt into meta for downstream metadata saving.
        meta: Dict[str, Any] = {}
        if isinstance(hit.get("meta"), dict):
            meta.update(hit["meta"])
        if isinstance(hit.get("metadata"), dict):
            meta.update(hit["metadata"])
        if isinstance(hit.get("prompt"), str) and "prompt" not in meta:
            meta["prompt"] = hit["prompt"]

        item: Dict[str, Any] = {
            "id": hit_id,
            "url": full_url,
            "type": hit.get("type") or "image",
            "nsfw": nsfw,
            "createdAt": hit.get("createdAt"),
            "meta": meta,
        }

        if isinstance(username, str) and username.strip():
            item["username"] = username.strip()

        # Best-effort compatibility with API items
        if isinstance(hit.get("width"), int):
            item["width"] = hit["width"]
        if isinstance(hit.get("height"), int):
            item["height"] = hit["height"]
        if isinstance(hit.get("postId"), int):
            item["postId"] = hit["postId"]

        return item

    @staticmethod
    def _extension_from_mime_type(mime_type: Any) -> Optional[str]:
        if not isinstance(mime_type, str):
            return None
        mt = mime_type.strip().lower()
        mapping = {
            "image/jpeg": ".jpeg",
            "image/jpg": ".jpeg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
        }
        return mapping.get(mt)

    @staticmethod
    def _default_extension_for_type(media_type: Any) -> str:
        mt = str(media_type or "").lower()
        if mt == "video":
            return ".mp4"
        return ".jpeg"


