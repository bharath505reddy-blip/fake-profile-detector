"""
live_enrichment/linkedin_enricher.py — LinkedIn profile enrichment.

LinkedIn's API is highly restrictive. This module:
  1. Attempts the LinkedIn v2 API if LINKEDIN_ACCESS_TOKEN is set.
  2. Falls back to accepting manually provided profile data dict.

The manual fallback allows the UI to pass form data and still compute features.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import requests as _http
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

_API_BASE = "https://api.linkedin.com/v2"
_TIMEOUT = 10

# Buzzword / spam headline keywords
_HEADLINE_BUZZWORDS = {
    "guru", "ninja", "visionary", "rockstar", "unicorn", "wizard",
    "evangelist", "thought leader", "growth hacker", "disruptor",
    "hustler", "maven", "mastermind", "trailblazer", "influencer",
}


def _connection_bucket(connections: int) -> int:
    """LinkedIn connection count bucket: 0=<50, 1=50-150, 2=150-500, 3=500+"""
    if connections >= 500:
        return 3
    if connections >= 150:
        return 2
    if connections >= 50:
        return 1
    return 0


def _headline_buzzword_score(headline: str) -> float:
    if not headline:
        return 0.0
    words = re.findall(r"\b\w+\b", headline.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _HEADLINE_BUZZWORDS)
    return hits / len(words)


def _career_consistency(positions: list) -> float:
    """
    Returns 1.0 if dates are sequential with no impossible overlaps, else 0.0-1.0.
    """
    if not positions or len(positions) < 2:
        return 1.0
    violations = 0
    for i in range(len(positions) - 1):
        try:
            end_i = positions[i].get("end_year") or 9999
            start_next = positions[i + 1].get("start_year") or 0
            if end_i < start_next - 1:
                violations += 1
        except Exception:
            pass
    return max(0.0, 1.0 - violations / max(len(positions) - 1, 1))


def _api_enrich(identifier: str) -> Optional[Dict]:
    """LinkedIn v2 API — requires access token."""
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
    if not token or not REQUESTS_AVAILABLE:
        return None
    try:
        headers = {"Authorization": f"Bearer {token}",
                   "X-Restli-Protocol-Version": "2.0.0"}
        r = _http.get(
            f"{_API_BASE}/me",
            headers=headers,
            params={"projection": "(id,firstName,lastName,headline,summary,"
                                   "numConnections,numConnectionsCapped)"},
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        logger.debug("LinkedIn API → %d", r.status_code)
    except Exception as exc:
        logger.debug("LinkedIn API error: %s", exc)
    return None


def enrich(identifier: str, manual_data: Optional[Dict] = None) -> Dict:
    """
    Enrich a LinkedIn profile.

    Args:
        identifier: Username or profile URL.
        manual_data: Pre-filled dict from manual form input (fallback).

    Returns standardised dict with raw_data, features, data_completeness_score.
    """
    api_data = _api_enrich(identifier)
    raw: Dict = api_data or manual_data or {}

    if not raw:
        return {
            "error": "No LinkedIn data available (no LINKEDIN_ACCESS_TOKEN and no manual data)",
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": "linkedin",
        }

    fields_fetched = 0
    fields_total = 6

    connections = raw.get("connections", raw.get("numConnections", 0)) or 0
    endorsements = raw.get("endorsements", 0) or 0
    skills = raw.get("skills", []) or []
    positions = raw.get("positions", []) or []
    education = raw.get("education", []) or []
    headline = raw.get("headline", "") or ""
    summary = raw.get("summary", raw.get("about", "")) or ""
    has_photo = bool(raw.get("profile_pic_url", raw.get("has_photo", False)))

    if connections:
        fields_fetched += 2
    if headline:
        fields_fetched += 1
    if summary:
        fields_fetched += 1
    if positions:
        fields_fetched += 1
    if education:
        fields_fetched += 1

    skill_count = len(skills)
    features = {
        "connections": connections,
        "connection_count_bucket": _connection_bucket(connections),
        "endorsement_to_skill_ratio": endorsements / max(skill_count, 1),
        "position_count": len(positions),
        "education_count": len(education),
        "profile_section_completeness": (
            int(bool(headline)) + int(bool(summary)) +
            int(bool(positions)) + int(bool(education)) +
            int(bool(skills)) + int(has_photo)
        ) / 6.0,
        "headline_keyword_spam_score": _headline_buzzword_score(headline),
        "career_trajectory_consistency": _career_consistency(positions),
        "skill_count": skill_count,
        "headline": headline,
        "bio": summary,
    }

    return {
        "platform": "linkedin",
        "username": identifier,
        "raw_data": raw,
        "features": features,
        "data_completeness_score": min(1.0, fields_fetched / fields_total),
    }
