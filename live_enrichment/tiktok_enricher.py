"""
live_enrichment/tiktok_enricher.py — TikTok profile enrichment.

TikTok's official API is heavily restricted. This module:
  1. Attempts TikTok Research API if TIKTOK_API_KEY is set.
  2. Falls back to accepting manually provided data.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import requests as _http
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

_TIMEOUT = 10


def _api_enrich(username: str) -> Optional[Dict]:
    """TikTok Research API (requires TIKTOK_API_KEY)."""
    key = os.environ.get("TIKTOK_API_KEY")
    if not key or not REQUESTS_AVAILABLE:
        return None
    try:
        r = _http.post(
            "https://open.tiktokapis.com/v2/research/user/info/",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={"username": username},
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json().get("data", {}).get("user_info", {})
        logger.debug("TikTok API → %d", r.status_code)
    except Exception as exc:
        logger.debug("TikTok API error: %s", exc)
    return None


def enrich(username: str, manual_data: Optional[Dict] = None) -> Dict:
    """
    Enrich a TikTok profile.

    Args:
        username: TikTok username.
        manual_data: Pre-filled dict from manual form input.
    """
    api_data = _api_enrich(username)
    raw: Dict = api_data or manual_data or {}

    if not raw:
        return {
            "error": "No TikTok data available (no TIKTOK_API_KEY and no manual data)",
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": "tiktok",
        }

    fields_fetched = 0
    fields_total = 4

    followers = raw.get("followers", raw.get("follower_count", 0)) or 0
    following = raw.get("following", raw.get("following_count", 0)) or 0
    video_count = raw.get("videos", raw.get("video_count", 0)) or 0
    likes_received = raw.get("likes_received", raw.get("total_favorited", 0)) or 0
    bio = raw.get("bio", raw.get("signature", "")) or ""
    verified = bool(raw.get("verified", raw.get("is_verified", False)))
    avatar_url = raw.get("avatar_url", raw.get("avatar_thumb", {}).get("url_list", [""])[0]) or ""

    if followers > 0:
        fields_fetched += 2
    if video_count > 0:
        fields_fetched += 1
    fields_fetched += 1

    avg_likes_per_video = likes_received / max(video_count, 1)

    features = {
        "followers": followers,
        "following": following,
        "videos": video_count,
        "bio": bio,
        "avatar_url": avatar_url,
        "is_verified": int(verified),
        "follower_following_ratio": followers / max(following, 1),
        "likes_to_follower_ratio": likes_received / max(followers, 1),
        "avg_likes_per_video": avg_likes_per_video,
        "bio_length": len(bio),
    }

    return {
        "platform": "tiktok",
        "username": username,
        "name": raw.get("display_name") or raw.get("nickname") or username,
        "followers": followers,
        "following": following,
        "posts": video_count,
        "bio": bio,
        "is_verified": verified,
        "has_avatar": bool(avatar_url),
        "avatar_url": avatar_url,
        "account_age_days": None,
        "location": None,
        "website": None,
        "completeness": min(1.0, fields_fetched / fields_total),
        "raw_data": raw,
        "features": features,
        "data_completeness_score": min(1.0, fields_fetched / fields_total),
        "total_videos": video_count,
        "total_likes": likes_received,
    }
