"""
live_enrichment/instagram_enricher.py — Instagram live profile enrichment.

Uses RapidAPI Instagram Scraper API (RAPIDAPI_INSTAGRAM_KEY env var).
Falls back to instaloader for public profiles if key unavailable.
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
    logger.debug("instaloader not available — RapidAPI-only mode.")

_RAPIDAPI_HOST = "instagram-scraper-api2.p.rapidapi.com"
_RAPIDAPI_BASE = f"https://{_RAPIDAPI_HOST}"
_TIMEOUT = 10


def _rapidapi_enrich(username: str) -> Optional[Dict]:
    """Fetch profile via RapidAPI Instagram Scraper API."""
    key = os.environ.get("RAPIDAPI_INSTAGRAM_KEY")
    if not key or not REQUESTS_AVAILABLE:
        return None

    try:
        r = _http.get(
            f"{_RAPIDAPI_BASE}/v1/info",
            params={"username_or_id_or_url": username},
            headers={
                "x-rapidapi-key": key,
                "x-rapidapi-host": _RAPIDAPI_HOST,
            },
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            user = data.get("data", {})
            if not user:
                logger.debug("RapidAPI: empty data for '%s'", username)
                return None
            return {
                "username": user.get("username", username),
                "followers": user.get("follower_count", 0),
                "following": user.get("following_count", 0),
                "media_count": user.get("media_count", 0),
                "bio": user.get("biography", ""),
                "profile_pic_url": user.get("profile_pic_url_hd") or user.get("profile_pic_url", ""),
                "is_verified": user.get("is_verified", False),
                "website": user.get("external_url", ""),
                "full_name": user.get("full_name", ""),
                "is_private": user.get("is_private", False),
                "is_business": user.get("is_business", False),
            }
        logger.debug("RapidAPI Instagram → %d: %s", r.status_code, r.text[:200])
    except Exception as exc:
        logger.debug("RapidAPI Instagram error for '%s': %s", username, exc)
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
            "full_name": profile.full_name or "",
            "is_private": profile.is_private,
            "is_business": profile.is_business_account,
        }
    except Exception as exc:
        logger.debug("instaloader error for '%s': %s", username, exc)
        return None


def enrich(username: str) -> Dict:
    """
    Fetch Instagram profile data and compute enrichment features.

    Returns a standardised dict with raw_data, features, data_completeness_score.
    """
    raw: Dict = {}
    fields_fetched = 0
    fields_total = 4

    raw = _rapidapi_enrich(username) or _instaloader_enrich(username) or {}

    if not raw:
        return {
            "error": f"Instagram profile '{username}' not accessible "
                     "(RapidAPI key missing/failed and instaloader failed)",
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": "instagram",
        }

    fields_fetched = 4

    followers = raw.get("followers", 0) or 0
    following = raw.get("following", 0) or 0
    media_count = raw.get("media_count", 0) or 0
    bio = raw.get("bio", "") or ""
    profile_pic = raw.get("profile_pic_url", "") or ""
    is_verified = bool(raw.get("is_verified", False))
    has_url = bool(raw.get("website", ""))
    is_private = bool(raw.get("is_private", False))
    is_business = bool(raw.get("is_business", False))

    features = {
        "followers": followers,
        "following": following,
        "posts": media_count,
        "bio": bio,
        "avatar_url": profile_pic,
        "is_verified": int(is_verified),
        "is_private": int(is_private),
        "is_business": int(is_business),
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
        "Instagram enrichment for '%s': completeness=%.2f followers=%d posts=%d verified=%s",
        username, fields_fetched / fields_total, followers, media_count, is_verified,
    )

    return {
        "platform": "instagram",
        "username": username,
        "name": raw.get("full_name", username),
        "followers": followers,
        "following": following,
        "posts": media_count,
        "bio": bio,
        "is_verified": is_verified,
        "has_avatar": bool(profile_pic),
        "avatar_url": profile_pic,
        "account_age_days": None,
        "location": None,
        "website": raw.get("website"),
        "completeness": min(1.0, fields_fetched / fields_total),
        "raw_data": raw,
        "features": features,
        "data_completeness_score": min(1.0, fields_fetched / fields_total),
        "media_count": media_count,
        "is_private": is_private,
        "is_business": is_business,
    }
