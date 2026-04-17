"""
features/cross_platform.py — Cross-platform identity verification.

Checks whether a given username exists consistently across multiple platforms,
comparing bios, avatars, and account ages to produce a consistency score.

All checks run in parallel with ThreadPoolExecutor. Per-platform calls are
cached in the CrossPlatformCache DB table (24-hour TTL).
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Platforms to check (ordered by API availability)
_CHECK_PLATFORMS = ["github", "reddit", "twitter", "youtube", "instagram", "linkedin", "tiktok"]
_PER_PLATFORM_TIMEOUT = 5.0  # seconds


# ---------------------------------------------------------------------------
# Text similarity helpers
# ---------------------------------------------------------------------------

def _cosine_sim_simple(a: str, b: str) -> float:
    """Simple word-frequency cosine similarity."""
    if not a or not b:
        return 0.0
    words_a = Counter(re.findall(r"\b\w+\b", a.lower()))
    words_b = Counter(re.findall(r"\b\w+\b", b.lower()))
    all_words = set(words_a) | set(words_b)
    if not all_words:
        return 0.0
    dot = sum(words_a.get(w, 0) * words_b.get(w, 0) for w in all_words)
    mag_a = math.sqrt(sum(v ** 2 for v in words_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in words_b.values()))
    if mag_a * mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


from collections import Counter


def _jaro_winkler_simple(s1: str, s2: str) -> float:
    """Simple Jaro-Winkler similarity (no external library required)."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    try:
        from rapidfuzz.distance import JaroWinkler
        return JaroWinkler.normalized_similarity(s1, s2)
    except ImportError:
        pass
    # Fallback: rough Jaro
    len_s1, len_s2 = len(s1), len(s2)
    match_dist = max(len_s1, len_s2) // 2 - 1
    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2
    matches = transpositions = 0
    for i, c in enumerate(s1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len_s2)
        for j in range(start, end):
            if not s2_matches[j] and c == s2[j]:
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0
    s1_trans = [s1[i] for i in range(len_s1) if s1_matches[i]]
    s2_trans = [s2[j] for j in range(len_s2) if s2_matches[j]]
    transpositions = sum(c1 != c2 for c1, c2 in zip(s1_trans, s2_trans)) // 2
    jaro = (matches / len_s1 + matches / len_s2 +
            (matches - transpositions) / matches) / 3
    # Winkler prefix bonus
    prefix = 0
    for c1, c2 in zip(s1[:4], s2[:4]):
        if c1 == c2:
            prefix += 1
        else:
            break
    return jaro + prefix * 0.1 * (1 - jaro)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_lookup(username: str, platform: str) -> Optional[Dict]:
    """Check CrossPlatformCache DB for a cached result."""
    try:
        from app import CrossPlatformCache, db
        cutoff = datetime.now(timezone.utc)
        row = CrossPlatformCache.query.filter_by(
            username=username, platform=platform
        ).first()
        if row and row.expires_at and row.expires_at.replace(tzinfo=timezone.utc) > cutoff:
            return json.loads(row.profile_data_json) if row.profile_data_json else {}
    except Exception:
        pass
    return None


def _cache_store(username: str, platform: str, exists: bool, profile_data: dict) -> None:
    """Store a result in CrossPlatformCache DB."""
    try:
        from app import CrossPlatformCache, db
        expires = datetime.now(timezone.utc) + timedelta(hours=24)
        row = CrossPlatformCache.query.filter_by(username=username, platform=platform).first()
        if row:
            row.exists = exists
            row.profile_data_json = json.dumps(profile_data)
            row.checked_at = datetime.now(timezone.utc)
            row.expires_at = expires
        else:
            row = CrossPlatformCache(
                username=username,
                platform=platform,
                exists=exists,
                profile_data_json=json.dumps(profile_data),
                checked_at=datetime.now(timezone.utc),
                expires_at=expires,
            )
            db.session.add(row)
        db.session.commit()
    except Exception as exc:
        logger.debug("CrossPlatformCache store failed: %s", exc)


# ---------------------------------------------------------------------------
# Per-platform existence check
# ---------------------------------------------------------------------------

def _check_platform(username: str, platform: str) -> Tuple[str, bool, Dict]:
    """
    Check if username exists on a given platform. Returns (platform, exists, data).
    Uses cache if available.
    """
    cached = _cache_lookup(username, platform)
    if cached is not None:
        return platform, bool(cached.get("exists", False)), cached

    try:
        from live_enrichment import enrich_profile
        result = enrich_profile(platform, username, timeout_seconds=_PER_PLATFORM_TIMEOUT)
        exists = "error" not in result or result.get("data_completeness_score", 0) > 0
        data = {
            "exists": exists,
            "features": result.get("features", {}),
            "raw_data": result.get("raw_data", {}),
            "completeness": result.get("data_completeness_score", 0.0),
        }
        _cache_store(username, platform, exists, data)
        return platform, exists, data
    except Exception as exc:
        logger.debug("Cross-platform check error [%s/%s]: %s", platform, username, exc)
        return platform, False, {"exists": False}


# ---------------------------------------------------------------------------
# Consistency scorers
# ---------------------------------------------------------------------------

def _bio_consistency(platform_data: Dict[str, Dict]) -> float:
    """Average pairwise cosine similarity of bios across platforms."""
    bios = []
    for plat, d in platform_data.items():
        features = d.get("features", {})
        raw = d.get("raw_data", {})
        bio = features.get("bio") or raw.get("bio") or raw.get("about") or raw.get("description") or ""
        if bio:
            bios.append(bio)
    if len(bios) < 2:
        return 1.0  # not enough data
    sims = []
    for i in range(len(bios)):
        for j in range(i + 1, len(bios)):
            sims.append(_cosine_sim_simple(bios[i], bios[j]))
    return sum(sims) / len(sims)


def _follower_consistency(platform_data: Dict[str, Dict]) -> float:
    """
    Check follower counts are in a similar log-scale range.
    Returns score 0-1 (1 = consistent).
    """
    follower_counts = []
    for d in platform_data.values():
        features = d.get("features", {})
        raw = d.get("raw_data", {})
        fc = features.get("followers") or raw.get("followers") or raw.get("subscribers") or 0
        if fc and fc > 0:
            follower_counts.append(math.log10(fc + 1))

    if len(follower_counts) < 2:
        return 1.0
    # Compute range of log follower counts
    log_range = max(follower_counts) - min(follower_counts)
    # > 3 orders of magnitude difference = suspicious
    return max(0.0, 1.0 - log_range / 5.0)


def _creation_date_consistency(platform_data: Dict[str, Dict]) -> float:
    """
    Check if creation dates across platforms are reasonably spaced.
    All created within same week = suspicious (score 0). Years apart = normal (score 1).
    """
    dates: List[float] = []
    for d in platform_data.values():
        features = d.get("features", {})
        raw = d.get("raw_data", {})
        age_days = features.get("account_age_days") or 0
        if age_days and age_days > 0:
            dates.append(float(age_days))

    if len(dates) < 2:
        return 1.0

    diffs = [abs(dates[i] - dates[j]) for i in range(len(dates)) for j in range(i + 1, len(dates))]
    min_diff = min(diffs)

    # < 7 days apart = suspicious, > 365 = fully natural
    score = min(1.0, min_diff / 365.0)
    return score


def _username_similarity(username: str, platform_data: Dict[str, Dict]) -> float:
    """Average Jaro-Winkler similarity of the queried username with any found usernames."""
    found_usernames = []
    for d in platform_data.values():
        raw = d.get("raw_data", {})
        found_uname = (raw.get("login") or raw.get("name") or raw.get("username") or "")
        if found_uname:
            found_usernames.append(found_uname.lower())
    if not found_usernames:
        return 1.0
    sims = [_jaro_winkler_simple(username.lower(), u) for u in found_usernames]
    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cross_platform_check(
    username: str,
    primary_platform: str,
    bio_text: Optional[str] = None,
    platforms_to_check: Optional[List[str]] = None,
) -> Dict:
    """
    Check a username's presence and consistency across multiple platforms.

    Args:
        username: The username to check.
        primary_platform: The platform where the profile was originally found.
        bio_text: Optional bio text from the primary platform.
        platforms_to_check: Specific platforms to check (default: all supported).

    Returns:
        Dict with keys:
            platforms_found_count, platforms_checked_count,
            cross_platform_presence_ratio, platform_list,
            bio_similarity_score, follower_consistency_score,
            creation_date_consistency, username_similarity_score,
            overall_consistency_score, raw_platform_data
    """
    check_platforms = [
        p for p in (_CHECK_PLATFORMS if platforms_to_check is None else platforms_to_check)
        if p != primary_platform
    ]

    platform_data: Dict[str, Dict] = {}
    found_platforms: List[str] = []

    t0 = time.time()
    # Parallel checks
    with ThreadPoolExecutor(max_workers=min(len(check_platforms), 4)) as ex:
        futures = {ex.submit(_check_platform, username, plat): plat for plat in check_platforms}
        for future in as_completed(futures, timeout=_PER_PLATFORM_TIMEOUT * 2):
            try:
                plat, exists, data = future.result(timeout=_PER_PLATFORM_TIMEOUT)
                if exists:
                    found_platforms.append(plat)
                    platform_data[plat] = data
            except Exception as exc:
                logger.debug("Cross-platform future error: %s", exc)

    elapsed = time.time() - t0
    platforms_checked = len(check_platforms)
    platforms_found = len(found_platforms)

    # Compute consistency scores
    bio_sim = _bio_consistency(platform_data)
    follower_consistency = _follower_consistency(platform_data)
    date_consistency = _creation_date_consistency(platform_data)
    username_sim = _username_similarity(username, platform_data)

    # Overall consistency = weighted average
    overall = (
        0.30 * bio_sim +
        0.25 * follower_consistency +
        0.25 * date_consistency +
        0.20 * username_sim
    ) if platform_data else 0.5

    logger.info(
        "Cross-platform check for '%s' [primary=%s]: found=%d/%d checked in %.1fs "
        "bio_sim=%.2f overall=%.2f",
        username, primary_platform, platforms_found, platforms_checked,
        elapsed, bio_sim, overall,
    )

    return {
        "platforms_found_count": platforms_found,
        "platforms_checked_count": platforms_checked,
        "cross_platform_presence_ratio": platforms_found / max(platforms_checked, 1),
        "platform_list": found_platforms,
        "bio_similarity_score": bio_sim,
        "follower_consistency_score": follower_consistency,
        "creation_date_consistency": date_consistency,
        "username_similarity_score": username_sim,
        "overall_consistency_score": overall,
        "raw_platform_data": {
            plat: {
                "features": d.get("features", {}),
                "completeness": d.get("completeness", 0),
            }
            for plat, d in platform_data.items()
        },
    }
