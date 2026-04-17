"""
live_enrichment/__init__.py — Router for live profile enrichment.

Usage:
    from live_enrichment import enrich_profile
    result = enrich_profile("github", "torvalds")
    # result = {
    #     "platform": "github",
    #     "username": "torvalds",
    #     "raw_data": {...},
    #     "features": {...},
    #     "data_completeness_score": 0.85,
    # }
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# API key env var names per platform (used by UI to show config status)
PLATFORM_ENV_VARS: Dict[str, str] = {
    "github":    "GITHUB_TOKEN",
    "reddit":    "",               # no key needed
    "twitter":   "TWITTER_BEARER_TOKEN",
    "instagram": "INSTAGRAM_ACCESS_TOKEN",
    "linkedin":  "LINKEDIN_ACCESS_TOKEN",
    "youtube":   "YOUTUBE_API_KEY",
    "discord":   "DISCORD_BOT_TOKEN",
    "tiktok":    "TIKTOK_API_KEY",
    "facebook":  "",               # no public API without user auth
    "snapchat":  "",               # no public API
}


def _lazy_import(platform: str):
    """Import the correct enricher module lazily."""
    try:
        if platform == "github":
            from live_enrichment import github_enricher
            return github_enricher
        elif platform == "reddit":
            from live_enrichment import reddit_enricher
            return reddit_enricher
        elif platform in ("twitter", "x"):
            from live_enrichment import twitter_enricher
            return twitter_enricher
        elif platform == "instagram":
            from live_enrichment import instagram_enricher
            return instagram_enricher
        elif platform == "linkedin":
            from live_enrichment import linkedin_enricher
            return linkedin_enricher
        elif platform == "youtube":
            from live_enrichment import youtube_enricher
            return youtube_enricher
        elif platform == "discord":
            from live_enrichment import discord_enricher
            return discord_enricher
        elif platform == "tiktok":
            from live_enrichment import tiktok_enricher
            return tiktok_enricher
    except ImportError as exc:
        logger.warning("Could not import enricher for '%s': %s", platform, exc)
    return None


def enrich_profile(
    platform: str,
    identifier: str,
    manual_data: Optional[Dict] = None,
    timeout_seconds: float = 15.0,
) -> Dict:
    """
    Route to the correct platform enricher and return a standardised result.

    Args:
        platform: Platform name (github, reddit, twitter, instagram, linkedin,
                  youtube, discord, tiktok, x).
        identifier: Username, user ID, or channel ID.
        manual_data: Optional fallback data dict (for platforms with no public API).
        timeout_seconds: Max seconds to wait for enrichment.

    Returns:
        Standardised dict:
          {
            platform, username, raw_data, features,
            data_completeness_score, error (optional)
          }
    """
    platform = platform.lower().strip()
    if platform == "x":
        platform = "twitter"

    mod = _lazy_import(platform)
    if mod is None:
        return {
            "error": f"No enricher available for platform '{platform}'",
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": platform,
        }

    def _run():
        # Most enrichers accept (identifier), some accept (identifier, manual_data)
        try:
            import inspect
            sig = inspect.signature(mod.enrich)
            if "manual_data" in sig.parameters:
                return mod.enrich(identifier, manual_data=manual_data)
            return mod.enrich(identifier)
        except Exception as exc:
            logger.warning("Enrichment error [%s/%s]: %s", platform, identifier, exc)
            return {
                "error": str(exc),
                "data_completeness_score": 0.0,
                "raw_data": {},
                "features": {},
                "platform": platform,
            }

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_run)
            return future.result(timeout=timeout_seconds)
    except FuturesTimeout:
        logger.warning("Enrichment timeout [%s/%s] after %.0fs", platform, identifier, timeout_seconds)
        return {
            "error": f"Enrichment timed out after {timeout_seconds}s",
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": platform,
        }
    except Exception as exc:
        logger.warning("Enrichment unexpected error [%s/%s]: %s", platform, identifier, exc)
        return {
            "error": str(exc),
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": platform,
        }


def get_platform_api_status() -> Dict[str, bool]:
    """
    Return dict of platform -> bool indicating whether an API key is configured.
    """
    return {
        platform: bool(os.environ.get(env_var))
        for platform, env_var in PLATFORM_ENV_VARS.items()
        if env_var  # empty env_var = no key needed = always "configured"
    }


def get_supported_platforms() -> list:
    """Return list of platforms that have enrichers implemented."""
    return list(PLATFORM_ENV_VARS.keys())


__all__ = [
    "enrich_profile",
    "get_platform_api_status",
    "get_supported_platforms",
    "PLATFORM_ENV_VARS",
]
