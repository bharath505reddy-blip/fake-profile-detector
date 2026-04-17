"""
live_enrichment/instagram_enricher.py — Instagram live profile enrichment.

Uses Instagram Basic Display API (INSTAGRAM_ACCESS_TOKEN env var).
Falls back to instaloader for public profiles if token unavailable.
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

try:
    import instaloader
    INSTALOADER_AVAILABLE = True
except ImportError:
    INSTALOADER_AVAILABLE = False
    logger.debug("instaloader not available — Instagram API-only mode.")

_GRAPH_BASE = "https://graph.instagram.com"
_TIMEOUT = 10


def _api_enrich(username: str) -> Optional[Dict]:
    """Use Instagram Basic Display API (requires access token)."""
    token = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
    if not token or not REQUESTS_AVAILABLE:
        return None

    try:
        # With Basic Display API, we can only look up the token owner's profile
        # For public lookups, Instagram Graph API Business requires page access
        r = _http.get(
            f"{_GRAPH_BASE}/me",
            params={"fields": "id,username,biography,followers_count,media_count,"
                               "profile_picture_url,website,is_verified",
                    "access_token": token},
            timeout=_TIMEOUT
        )
        if r.status_code == 200:
            return r.json()
        logger.debug("Instagram API → %d", r.status_code)
    except Exception as exc:
        logger.debug("Instagram API error: %s", exc)
    return None


def _instaloader_enrich(username: str) -> Optional[Dict]:
    """Use instaloader to fetch public Instagram profile data."""
    if not INSTALOADER_AVAILABLE:
        return None
    try:
        L = instaloader.Instaloader(quiet=True, download_pictures=False,
                                    download_videos=False, download_geotags=False,
                                    download_comments=False, save_metadata=False)
        profile = instaloader.Profile.from_username(L.context, username)
        return {
            "username": profile.username,
            "followers": profile.followers,
            "following": profile.followees,
            "media_count": profile.mediacount,
            "bio": profile.biography or "",
            "profile_pic_url": profile.profile_pic_url or "",
            "is_verified": profile.is_verified,
            "website": profile.external_url or "",
        }
    except Exception as exc:
        logger.debug("instaloader error for '%s': %s", username, exc)
        return None


def enrich(username: str) -> Dict:
    """
    Fetch Instagram profile data and compute enrichment features.

    Returns a standardised dict with raw_data, features, data_completeness_score.
    """
    # Try API first, then instaloader
    api_data = _api_enrich(username)
    raw: Dict = {}
    fields_fetched = 0
    fields_total = 4

    if api_data:
        raw = api_data
        fields_fetched = 3
    else:
        il_data = _instaloader_enrich(username)
        if il_data:
            raw = il_data
            fields_fetched = 3
        else:
            return {
                "error": f"Instagram profile '{username}' not accessible "
                         "(no INSTAGRAM_ACCESS_TOKEN and instaloader failed)",
                "data_completeness_score": 0.0,
                "raw_data": {},
                "features": {},
                "platform": "instagram",
            }

    followers = raw.get("followers", raw.get("followers_count", 0)) or 0
    following = raw.get("following", raw.get("following_count", 0)) or 0
    media_count = raw.get("media_count", 0) or 0
    bio = raw.get("bio", raw.get("biography", "")) or ""
    profile_pic = raw.get("profile_pic_url", raw.get("profile_picture_url", "")) or ""
    is_verified = bool(raw.get("is_verified", False))
    has_url = bool(raw.get("website", ""))
    fields_fetched += 1

    features = {
        "followers": followers,
        "following": following,
        "posts": media_count,
        "bio": bio,
        "avatar_url": profile_pic,
        "is_verified": int(is_verified),
        "follower_following_ratio": followers / max(following, 1),
        "media_per_follower": media_count / max(followers, 1),
        "bio_url_present": int(has_url),
        "bio_length": len(bio),
        "profile_completeness": (
            int(bool(bio)) + int(bool(profile_pic)) +
            int(has_url) + int(is_verified)
        ) / 4.0,
    }

    logger.info(
        "Instagram enrichment for '%s': completeness=%.2f followers=%d posts=%d",
        username, fields_fetched / fields_total, followers, media_count,
    )

    return {
        "platform": "instagram",
        "username": username,
        "raw_data": raw,
        "features": features,
        "data_completeness_score": min(1.0, fields_fetched / fields_total),
    }
