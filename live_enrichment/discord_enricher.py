"""
live_enrichment/discord_enricher.py — Discord profile enrichment.

Integrates with the existing Discord_scrapper module. Since Discord's API
requires bot tokens and user data is not publicly available, this module
primarily computes features from scraped/manually provided data.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import requests as _http
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

_DISCORD_API = "https://discord.com/api/v10"
_TIMEOUT = 8


def _get_user(user_id: str) -> Optional[Dict]:
    """Fetch Discord user data via Bot API (requires DISCORD_BOT_TOKEN)."""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token or not REQUESTS_AVAILABLE:
        return None
    try:
        r = _http.get(
            f"{_DISCORD_API}/users/{user_id}",
            headers={"Authorization": f"Bot {token}"},
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        logger.debug("Discord API → %d", r.status_code)
        return None
    except Exception as exc:
        logger.debug("Discord API error: %s", exc)
        return None


def _discord_id_to_creation(snowflake: str) -> Optional[datetime]:
    """Extract account creation timestamp from Discord snowflake ID."""
    try:
        snowflake_int = int(snowflake)
        DISCORD_EPOCH = 1420070400000  # Discord epoch in ms
        ts_ms = (snowflake_int >> 22) + DISCORD_EPOCH
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    except Exception:
        return None


def enrich(user_id_or_username: str, manual_data: Optional[Dict] = None) -> Dict:
    """
    Enrich a Discord profile with features.

    Args:
        user_id_or_username: Discord user ID (snowflake) or username.
        manual_data: Pre-filled dict from scraper or manual input.
    """
    raw: Dict = manual_data or {}
    api_data = _get_user(user_id_or_username) if user_id_or_username.isdigit() else None

    if api_data:
        raw.update(api_data)

    if not raw:
        return {
            "error": "No Discord data available (no DISCORD_BOT_TOKEN and no manual data)",
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": "discord",
        }

    fields_fetched = 0
    fields_total = 4

    username = raw.get("username", user_id_or_username)
    has_avatar = bool(raw.get("avatar") or raw.get("has_avatar"))
    account_age_days = raw.get("account_age_days", 0) or 0
    server_count = raw.get("server_count", raw.get("guild_count", 0)) or 0
    role_count = raw.get("role_count", 0) or 0
    message_count = raw.get("message_count", 0) or 0
    suspicion_score = raw.get("suspicion_score", 0.0) or 0.0

    # Try to derive account age from snowflake ID
    if account_age_days == 0 and user_id_or_username.isdigit():
        created = _discord_id_to_creation(user_id_or_username)
        if created:
            account_age_days = (datetime.now(timezone.utc) - created).days
            fields_fetched += 1

    if account_age_days > 0:
        fields_fetched += 1
    if server_count > 0:
        fields_fetched += 1
    fields_fetched += 1  # base data

    # Creation to first message gap (if provided by scraper)
    creation_gap = raw.get("creation_to_first_message_gap", None)

    message_frequency = message_count / max(account_age_days, 1) if account_age_days else 0

    features = {
        "username": username,
        "has_avatar": int(has_avatar),
        "account_age_days": account_age_days,
        "suspicion_score": suspicion_score,
        "server_count": server_count,
        "role_count": role_count,
        "message_frequency": message_frequency,
    }

    if creation_gap is not None:
        features["creation_to_first_message_gap"] = creation_gap

    return {
        "platform": "discord",
        "username": username,
        "raw_data": raw,
        "features": features,
        "data_completeness_score": min(1.0, fields_fetched / fields_total),
    }
