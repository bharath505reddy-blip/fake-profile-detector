"""
features/temporal.py — Temporal and behavioral pattern feature extraction.

All functions accept a profile_data dict and return a dict of feature_name -> float.
Missing input data is handled gracefully — returns 0.0 for any unavailable feature.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from dateutil import parser as dateutil_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False


def _parse_timestamps(raw: list) -> List[float]:
    """
    Parse a list of timestamps (ISO strings or unix floats) into sorted unix float seconds.
    Returns empty list if parsing fails.
    """
    result = []
    for t in raw:
        if t is None:
            continue
        try:
            result.append(float(t))
            continue
        except (TypeError, ValueError):
            pass
        if DATEUTIL_AVAILABLE:
            try:
                dt = dateutil_parser.parse(str(t))
                result.append(dt.timestamp())
                continue
            except Exception:
                pass
    return sorted(result)


def _burst_score(timestamps: List[float], burst_window_sec: float = 3600.0, burst_min_posts: int = 10) -> float:
    """
    Compute ratio of posts occurring in burst windows (>=burst_min_posts within burst_window_sec).
    Bots often post in tight clusters.
    Returns 0-1 float.
    """
    if len(timestamps) < burst_min_posts:
        return 0.0
    burst_post_set = set()
    n = len(timestamps)
    for i in range(n):
        count = 1
        window_members = [i]
        for j in range(i + 1, n):
            if timestamps[j] - timestamps[i] <= burst_window_sec:
                count += 1
                window_members.append(j)
            else:
                break
        if count >= burst_min_posts:
            burst_post_set.update(window_members)
    return round(len(burst_post_set) / n, 4)


def _posting_regularity_score(timestamps: List[float]) -> float:
    """
    Measure how regular post intervals are. Bots have near-zero std deviation (perfectly regular).
    Returns value 0-1 where higher = more regular (more bot-like).
    Formula: 1 / (1 + std_dev_hours)
    """
    if len(timestamps) < 2:
        return 0.0
    gaps_hours = [(timestamps[i + 1] - timestamps[i]) / 3600.0 for i in range(len(timestamps) - 1)]
    std_dev = float(np.std(gaps_hours))
    return round(1.0 / (1.0 + std_dev), 4)


def _time_of_day_entropy(timestamps: List[float]) -> float:
    """
    Compute Shannon entropy of post hour-of-day distribution (0-23 buckets), normalized by log(24).
    Low entropy = bot-like (posts concentrated in few hours).
    Returns 0-1 float.
    """
    if not timestamps:
        return 0.0
    try:
        from datetime import datetime, timezone
        hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in timestamps]
        counts = np.zeros(24)
        for h in hours:
            counts[h] += 1
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        entropy = -float(np.sum(probs * np.log(probs + 1e-9)))
        return round(entropy / math.log(24), 4)
    except Exception as exc:
        logger.debug("time_of_day_entropy error: %s", exc)
        return 0.0


def _active_hours_count(timestamps: List[float]) -> int:
    """
    Count distinct hours of day (0-23) with at least one post.
    Low count = suspicious.
    """
    if not timestamps:
        return 0
    try:
        from datetime import datetime, timezone
        hours = {datetime.fromtimestamp(t, tz=timezone.utc).hour for t in timestamps}
        return len(hours)
    except Exception:
        return 0


def _weekend_weekday_ratio(timestamps: List[float]) -> float:
    """
    Ratio of weekend posts (Sat=5, Sun=6) to weekday posts (Mon-Fri).
    Bots often post uniformly ignoring weekends.
    Returns float (0 if no weekday posts).
    """
    if not timestamps:
        return 0.0
    try:
        from datetime import datetime, timezone
        weekday_count = 0
        weekend_count = 0
        for t in timestamps:
            wd = datetime.fromtimestamp(t, tz=timezone.utc).weekday()
            if wd >= 5:
                weekend_count += 1
            else:
                weekday_count += 1
        if weekday_count == 0:
            return float(weekend_count) if weekend_count > 0 else 0.0
        return round(weekend_count / weekday_count, 4)
    except Exception:
        return 0.0


def _night_post_ratio(timestamps: List[float]) -> float:
    """
    Ratio of posts between 0:00 and 5:00 UTC.
    Bots often post at night (no human behavior).
    Returns 0-1 float.
    """
    if not timestamps:
        return 0.0
    try:
        from datetime import datetime, timezone
        night_count = sum(
            1 for t in timestamps
            if datetime.fromtimestamp(t, tz=timezone.utc).hour <= 5
        )
        return round(night_count / len(timestamps), 4)
    except Exception:
        return 0.0


def _follow_velocity(total_follows: int, account_age_days: float) -> float:
    """
    Average follows per day since account creation.
    High velocity = bot-like behavior.
    """
    if account_age_days <= 0:
        return float(total_follows)
    return round(total_follows / account_age_days, 4)


def _unfollow_rate(total_unfollows: int, total_follows: int) -> float:
    """
    Ratio of unfollowed to followed accounts.
    High unfollow rate = churn bot behavior.
    Returns 0-1.
    """
    if total_follows <= 0:
        return 0.0
    return round(min(total_unfollows / total_follows, 1.0), 4)


def _engagement_received_vs_given(
    likes_received: int, comments_received: int,
    likes_given: int, comments_given: int
) -> float:
    """
    (likes+comments received) / (likes+comments given).
    Low ratio = fake (fakes give much more than they receive).
    """
    given = likes_given + comments_given
    received = likes_received + comments_received
    if given <= 0:
        return float(received) if received > 0 else 1.0
    return round(received / given, 4)


def _reply_to_original_ratio(reply_count: int, original_post_count: int) -> float:
    """
    Ratio of replies/retweets to original content.
    Bots are heavy retweeters — high ratio = suspicious.
    """
    if original_post_count <= 0:
        return float(reply_count) if reply_count > 0 else 0.0
    return round(reply_count / original_post_count, 4)


def _activity_gap_max(timestamps: List[float]) -> float:
    """
    Longest period of inactivity between consecutive posts in days.
    Returns 0 if fewer than 2 posts.
    """
    if len(timestamps) < 2:
        return 0.0
    gaps_days = [(timestamps[i + 1] - timestamps[i]) / 86400.0 for i in range(len(timestamps) - 1)]
    return round(max(gaps_days), 4)


def _activity_gap_variance(timestamps: List[float]) -> float:
    """
    Variance of inter-post gaps in hours.
    Very low variance = bot-like regular posting.
    """
    if len(timestamps) < 2:
        return 0.0
    gaps_hours = [(timestamps[i + 1] - timestamps[i]) / 3600.0 for i in range(len(timestamps) - 1)]
    return round(float(np.var(gaps_hours)), 4)


def _account_age_vs_activity(total_posts: int, account_age_days: float) -> float:
    """
    Posts per day normalized by account age.
    New accounts with very high post rate = suspicious.
    """
    if account_age_days <= 0:
        return float(total_posts)
    return round(total_posts / account_age_days, 4)


def extract_temporal_features(profile_data: dict) -> dict:
    """
    Extract all temporal and behavioral features from a profile data dict.

    Expected profile_data keys (all optional):
      post_timestamps        — list of ISO strings or unix timestamps
      follow_timestamps      — list of timestamps for follows
      account_created_at     — account creation timestamp (ISO or unix)
      total_follows          — int
      total_unfollows        — int
      likes_given            — int
      comments_given         — int
      likes_received         — int
      comments_received      — int
      reply_count            — int
      original_post_count    — int
      account_age_days       — float (will be computed from account_created_at if not provided)
      total_posts            — int (fallback: len(post_timestamps))

    Returns dict mapping feature name -> float.
    All values are guaranteed to be finite floats (NaN replaced with 0.0).
    """
    features: Dict[str, float] = {}

    try:
        raw_ts = profile_data.get("post_timestamps") or []
        timestamps = _parse_timestamps(raw_ts) if isinstance(raw_ts, list) else []

        # Account age
        account_age_days: float = float(profile_data.get("account_age_days") or 0)
        if account_age_days <= 0:
            raw_created = profile_data.get("account_created_at")
            if raw_created:
                created_ts = _parse_timestamps([raw_created])
                if created_ts:
                    import time
                    account_age_days = max((time.time() - created_ts[0]) / 86400.0, 1.0)

        total_posts = int(profile_data.get("total_posts") or len(timestamps))
        total_follows = int(profile_data.get("total_follows") or 0)
        total_unfollows = int(profile_data.get("total_unfollows") or 0)
        likes_given = int(profile_data.get("likes_given") or 0)
        comments_given = int(profile_data.get("comments_given") or 0)
        likes_received = int(profile_data.get("likes_received") or 0)
        comments_received = int(profile_data.get("comments_received") or 0)
        reply_count = int(profile_data.get("reply_count") or 0)
        original_post_count = int(profile_data.get("original_post_count") or max(total_posts - reply_count, 1))

        features["burst_score"] = _burst_score(timestamps)
        features["posting_regularity_score"] = _posting_regularity_score(timestamps)
        features["time_of_day_entropy"] = _time_of_day_entropy(timestamps)
        features["active_hours_count"] = float(_active_hours_count(timestamps))
        features["weekend_weekday_ratio"] = _weekend_weekday_ratio(timestamps)
        features["night_post_ratio"] = _night_post_ratio(timestamps)
        features["follow_velocity"] = _follow_velocity(total_follows, account_age_days)
        features["unfollow_rate"] = _unfollow_rate(total_unfollows, total_follows)
        features["engagement_received_vs_given"] = _engagement_received_vs_given(
            likes_received, comments_received, likes_given, comments_given
        )
        features["reply_to_original_ratio"] = _reply_to_original_ratio(reply_count, original_post_count)
        features["activity_gap_max"] = _activity_gap_max(timestamps)
        features["activity_gap_variance"] = _activity_gap_variance(timestamps)
        features["account_age_vs_activity"] = _account_age_vs_activity(total_posts, account_age_days)

    except Exception as exc:
        logger.warning("extract_temporal_features error: %s", exc)

    # Replace NaN/inf with 0.0
    for k, v in features.items():
        if not math.isfinite(v):
            features[k] = 0.0

    return features
