"""
live_enrichment/twitter_enricher.py — Twitter/X live profile enrichment.

Requires TWITTER_BEARER_TOKEN environment variable for Twitter API v2.
If absent, returns gracefully with data_completeness_score=0.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests as _http
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

_API_BASE = "https://api.twitter.com/2"
_TIMEOUT = 10


def _bearer_headers() -> Optional[dict]:
    token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _get(path: str, params: dict = None) -> Optional[dict]:
    headers = _bearer_headers()
    if headers is None:
        return None
    if not REQUESTS_AVAILABLE:
        return None
    try:
        r = _http.get(f"{_API_BASE}{path}", headers=headers,
                      params=params or {}, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        logger.debug("Twitter API %s → %d", path, r.status_code)
        return None
    except Exception as exc:
        logger.debug("Twitter API error: %s", exc)
        return None


def _count_in_tweets(tweets: List[dict], key: str) -> int:
    count = 0
    for t in tweets:
        entities = t.get("entities", {}) or {}
        count += len(entities.get(key, []))
    return count


def enrich(username: str) -> Dict:
    """
    Fetch Twitter/X profile data and compute enrichment features.
    Requires TWITTER_BEARER_TOKEN env var.
    """
    if not os.environ.get("TWITTER_BEARER_TOKEN"):
        return {
            "error": "TWITTER_BEARER_TOKEN not configured",
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": "twitter",
        }

    if not REQUESTS_AVAILABLE:
        return {"error": "requests not installed", "data_completeness_score": 0.0,
                "raw_data": {}, "features": {}, "platform": "twitter"}

    fields_fetched = 0
    fields_total = 4

    # ---- User lookup ----
    user_resp = _get(
        f"/users/by/username/{username}",
        params={
            "user.fields": (
                "created_at,description,entities,id,location,name,pinned_tweet_id,"
                "profile_image_url,protected,public_metrics,url,verified,withheld"
            )
        }
    )

    if user_resp is None or "data" not in user_resp:
        return {"error": f"Twitter user '{username}' not found or API error",
                "data_completeness_score": 0.0, "raw_data": {}, "features": {},
                "platform": "twitter"}

    udata = user_resp["data"]
    fields_fetched += 1

    metrics = udata.get("public_metrics", {}) or {}
    followers = metrics.get("followers_count", 0) or 0
    following = metrics.get("following_count", 0) or 0
    tweet_count = metrics.get("tweet_count", 0) or 0
    listed_count = metrics.get("listed_count", 0) or 0
    created_at_str = udata.get("created_at", "")
    description = udata.get("description") or ""
    profile_image = udata.get("profile_image_url", "") or ""
    verified = bool(udata.get("verified", False))

    account_age_days = 0.0
    if created_at_str:
        try:
            created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            account_age_days = (datetime.now(timezone.utc) - created).days
            fields_fetched += 1
        except Exception:
            pass

    # Check for default avatar
    is_default_avatar = (
        "default_profile_images" in profile_image or not profile_image
    )

    # ---- Recent tweets ----
    user_id = udata.get("id", "")
    tweets: List[dict] = []
    if user_id:
        tweets_resp = _get(
            f"/users/{user_id}/tweets",
            params={
                "max_results": 50,
                "tweet.fields": "created_at,entities,public_metrics,referenced_tweets",
                "expansions": "referenced_tweets.id",
            }
        )
        if tweets_resp and "data" in tweets_resp:
            tweets = tweets_resp["data"]
            fields_fetched += 1

    total_tweets = max(len(tweets), 1)
    retweets = sum(1 for t in tweets if any(
        (r.get("type") == "retweeted") for r in (t.get("referenced_tweets") or [])
    ))
    replies = sum(1 for t in tweets if any(
        (r.get("type") == "replied_to") for r in (t.get("referenced_tweets") or [])
    ))
    urls_in_tweets = sum(
        1 for t in tweets if (t.get("entities") or {}).get("urls")
    )
    hashtag_total = _count_in_tweets(tweets, "hashtags")
    mention_total = _count_in_tweets(tweets, "mentions")
    avg_tweet_len = sum(len(t.get("text", "")) for t in tweets) / total_tweets

    timestamps: List[float] = []
    for t in tweets:
        ts_str = t.get("created_at", "")
        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                timestamps.append(dt.timestamp())
            except Exception:
                pass

    fields_fetched += 1

    features = {
        "followers": followers,
        "following": following,
        "tweets": tweet_count,
        "listed_count": listed_count,
        "account_age_days": account_age_days,
        "bio": description,
        "avatar_url": profile_image,
        "follower_following_ratio": followers / max(following, 1),
        "listed_ratio": listed_count / max(followers, 1),
        "tweet_frequency": tweet_count / max(account_age_days, 1),
        "retweet_ratio": retweets / total_tweets,
        "reply_ratio": replies / total_tweets,
        "avg_tweet_length": avg_tweet_len,
        "hashtag_density": hashtag_total / total_tweets,
        "mention_density": mention_total / total_tweets,
        "url_in_tweet_ratio": urls_in_tweets / total_tweets,
        "default_avatar": int(is_default_avatar),
        "has_description": int(bool(description.strip())),
        "description_length": len(description),
        "is_verified": int(verified),
    }

    logger.info(
        "Twitter enrichment for '@%s': completeness=%.2f followers=%d tweets=%d",
        username, fields_fetched / fields_total, followers, tweet_count,
    )

    return {
        "platform": "twitter",
        "username": username,
        "name": udata.get("name", ""),
        "followers": followers,
        "following": following,
        "posts": tweet_count,
        "bio": description,
        "is_verified": verified,
        "has_avatar": not is_default_avatar,
        "avatar_url": profile_image,
        "account_age_days": int(account_age_days) if account_age_days else None,
        "location": udata.get("location"),
        "website": udata.get("url"),
        "completeness": min(1.0, fields_fetched / fields_total),
        "raw_data": {
            "username": udata.get("username", ""),
            "name": udata.get("name", ""),
            "followers": followers,
            "following": following,
            "tweet_count": tweet_count,
            "listed_count": listed_count,
            "created_at": created_at_str,
            "description": description,
            "profile_image_url": profile_image,
            "verified": verified,
            "location": udata.get("location"),
            "url": udata.get("url"),
        },
        "features": features,
        "data_completeness_score": min(1.0, fields_fetched / fields_total),
        "tweets": tweet_count,
        "listed_count": listed_count,
        "posts_list": tweets,
        "timestamps": timestamps,
    }
