"""
live_enrichment/youtube_enricher.py — YouTube channel enrichment.

Requires YOUTUBE_API_KEY environment variable (YouTube Data API v3).
"""
from __future__ import annotations

import logging
import math
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests as _http
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

_API_BASE = "https://www.googleapis.com/youtube/v3"
_TIMEOUT = 10


def _get(endpoint: str, params: dict) -> Optional[dict]:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return None
    if not REQUESTS_AVAILABLE:
        return None
    params["key"] = key
    try:
        r = _http.get(f"{_API_BASE}/{endpoint}", params=params, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        logger.debug("YouTube API %s → %d", endpoint, r.status_code)
        return None
    except Exception as exc:
        logger.debug("YouTube API error: %s", exc)
        return None


def _upload_gaps(publish_dates: List[str]) -> tuple[float, float]:
    """Returns (mean_gap_days, std_gap_days) from ISO date strings."""
    if len(publish_dates) < 2:
        return 0.0, 0.0
    try:
        dts = sorted(
            datetime.fromisoformat(d.replace("Z", "+00:00"))
            for d in publish_dates if d
        )
        gaps = [(dts[i + 1] - dts[i]).days for i in range(len(dts) - 1)]
        if not gaps:
            return 0.0, 0.0
        mean = sum(gaps) / len(gaps)
        std = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
        return mean, std
    except Exception:
        return 0.0, 0.0


def enrich(channel_id_or_username: str) -> Dict:
    """
    Fetch YouTube channel data and compute enrichment features.
    Accepts a channel ID (@handle or UC... ID).
    Requires YOUTUBE_API_KEY.
    """
    if not os.environ.get("YOUTUBE_API_KEY"):
        return {
            "error": "YOUTUBE_API_KEY not configured",
            "data_completeness_score": 0.0,
            "raw_data": {},
            "features": {},
            "platform": "youtube",
        }

    if not REQUESTS_AVAILABLE:
        return {"error": "requests not installed", "data_completeness_score": 0.0,
                "raw_data": {}, "features": {}, "platform": "youtube"}

    fields_fetched = 0
    fields_total = 4

    # ---- Channel lookup ----
    # Try by handle first, then by channel ID
    identifier = channel_id_or_username.lstrip("@")
    ch_resp = _get("channels", {
        "part": "snippet,statistics,contentDetails",
        "forHandle": identifier,
    })
    if not ch_resp or not ch_resp.get("items"):
        ch_resp = _get("channels", {
            "part": "snippet,statistics,contentDetails",
            "id": channel_id_or_username,
        })

    if not ch_resp or not ch_resp.get("items"):
        return {"error": f"YouTube channel '{channel_id_or_username}' not found",
                "data_completeness_score": 0.0, "raw_data": {}, "features": {},
                "platform": "youtube"}

    ch = ch_resp["items"][0]
    fields_fetched += 1
    snippet = ch.get("snippet", {})
    stats = ch.get("statistics", {})
    content_details = ch.get("contentDetails", {})

    channel_title = snippet.get("title", "") or ""
    description = snippet.get("description", "") or ""
    created_at_str = snippet.get("publishedAt", "") or ""
    subscribers = int(stats.get("subscriberCount", 0) or 0)
    video_count = int(stats.get("videoCount", 0) or 0)
    view_count = int(stats.get("viewCount", 0) or 0)
    uploads_playlist = content_details.get("relatedPlaylists", {}).get("uploads", "")

    account_age_days = 0.0
    if created_at_str:
        try:
            created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            account_age_days = (datetime.now(timezone.utc) - created).days
            fields_fetched += 1
        except Exception:
            pass

    # ---- Recent videos ----
    videos: List[dict] = []
    publish_dates: List[str] = []
    desc_lengths: List[int] = []
    tags_per_video: List[int] = []
    comments_disabled_count = 0

    if uploads_playlist:
        playlist_resp = _get("playlistItems", {
            "part": "snippet",
            "playlistId": uploads_playlist,
            "maxResults": 20,
        })
        if playlist_resp and playlist_resp.get("items"):
            fields_fetched += 1
            video_ids = [
                item["snippet"]["resourceId"]["videoId"]
                for item in playlist_resp["items"]
                if item.get("snippet", {}).get("resourceId", {}).get("videoId")
            ]
            # Batch fetch video details
            if video_ids:
                vids_resp = _get("videos", {
                    "part": "snippet,statistics",
                    "id": ",".join(video_ids),
                })
                if vids_resp and vids_resp.get("items"):
                    fields_fetched += 1
                    for v in vids_resp["items"]:
                        vsnip = v.get("snippet", {})
                        vstats = v.get("statistics", {})
                        pub = vsnip.get("publishedAt", "")
                        if pub:
                            publish_dates.append(pub)
                        desc = vsnip.get("description", "") or ""
                        desc_lengths.append(len(desc.split()))
                        tags = vsnip.get("tags") or []
                        tags_per_video.append(len(tags))
                        if vstats.get("commentCount") is None:
                            comments_disabled_count += 1
                        videos.append(v)

    total_videos = max(video_count, 1)
    mean_gap, std_gap = _upload_gaps(publish_dates)

    features = {
        "channel_name": channel_title,
        "subscribers": subscribers,
        "videos": video_count,
        "about": description,
        "account_age_days": account_age_days,
        "subscriber_to_video_ratio": subscribers / max(video_count, 1),
        "avg_views_per_video": view_count / max(video_count, 1),
        "upload_frequency_days": mean_gap,
        "upload_regularity": std_gap,
        "view_to_subscriber_ratio": (view_count / video_count) / max(subscribers, 1),
        "description_length_avg": sum(desc_lengths) / max(len(desc_lengths), 1),
        "tags_per_video_avg": sum(tags_per_video) / max(len(tags_per_video), 1),
        "comment_disabled_ratio": comments_disabled_count / max(len(videos), 1),
    }

    logger.info(
        "YouTube enrichment for '%s': completeness=%.2f subs=%d videos=%d",
        channel_id_or_username, fields_fetched / fields_total, subscribers, video_count,
    )

    thumbnail_url = snippet.get("thumbnails", {}).get("default", {}).get("url") or None

    return {
        "platform": "youtube",
        "username": channel_id_or_username,
        "name": channel_title,
        "followers": subscribers,
        "following": None,
        "posts": video_count,
        "bio": description,
        "about": description,
        "is_verified": False,
        "has_avatar": bool(thumbnail_url),
        "avatar_url": thumbnail_url,
        "account_age_days": int(account_age_days) if account_age_days else None,
        "location": None,
        "website": None,
        "completeness": min(1.0, fields_fetched / fields_total),
        "raw_data": {
            "channel_title": channel_title,
            "subscribers": subscribers,
            "video_count": video_count,
            "view_count": view_count,
            "created_at": created_at_str,
            "description": description,
        },
        "features": features,
        "data_completeness_score": min(1.0, fields_fetched / fields_total),
        "channel_name": channel_title,
        "subscribers": subscribers,
        "total_videos": video_count,
        "view_count": view_count,
    }
