"""
live_enrichment/github_enricher.py — GitHub live profile enrichment.

Fetches public GitHub profile data and computes ML-ready features.
Uses GITHUB_TOKEN env var if set (increases rate limit from 60→5000 req/hr).
"""
from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests as _http
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

_BASE = "https://api.github.com"
_TIMEOUT = 8


def _headers() -> dict:
    h = {"User-Agent": "FakeProfileDetector/1.0",
         "Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        h["Authorization"] = f"token {token}"
    return h


def _get(path: str) -> Optional[dict]:
    if not REQUESTS_AVAILABLE:
        return None
    try:
        r = _http.get(f"{_BASE}{path}", headers=_headers(), timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        logger.debug("GitHub API %s → %d", path, r.status_code)
        return None
    except Exception as exc:
        logger.debug("GitHub API error %s: %s", path, exc)
        return None


def _shannon_entropy(values: List[float]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    return -sum((v / total) * math.log2(v / total) for v in values if v > 0)


def enrich(username: str) -> Dict:
    """
    Fetch GitHub profile data and compute enrichment features.

    Returns a standardised dict with:
      raw_data, features, data_completeness_score, error (optional)
    """
    if not REQUESTS_AVAILABLE:
        return {"error": "requests not installed", "data_completeness_score": 0.0,
                "raw_data": {}, "features": {}}

    raw: Dict = {}
    features: Dict = {}
    fields_fetched = 0
    fields_total = 8

    # ---- User profile ----
    user = _get(f"/users/{username}")
    if user is None:
        return {"error": f"GitHub user '{username}' not found",
                "data_completeness_score": 0.0, "raw_data": {}, "features": {}}

    raw.update(user)
    fields_fetched += 1

    followers = user.get("followers", 0) or 0
    following = user.get("following", 0) or 0
    public_repos = user.get("public_repos", 0) or 0
    public_gists = user.get("public_gists", 0) or 0
    created_at_str = user.get("created_at", "")
    avatar_url = user.get("avatar_url", "")
    bio = user.get("bio") or ""
    name = user.get("name") or ""

    # Account age
    account_age_days = 0.0
    if created_at_str:
        try:
            created = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            account_age_days = (datetime.now(timezone.utc) - created).days
            fields_fetched += 1
        except Exception:
            pass

    # ---- Repos ----
    repos_data = _get(f"/users/{username}/repos?per_page=100&sort=updated") or []
    if repos_data:
        fields_fetched += 1

    languages: Dict[str, int] = {}
    stars_total = 0
    forks_count = 0
    repos_with_readme = 0  # approximated by description presence
    for repo in repos_data:
        if isinstance(repo, dict):
            stars_total += repo.get("stargazers_count", 0) or 0
            if repo.get("fork"):
                forks_count += 1
            lang = repo.get("language")
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
            if repo.get("description"):
                repos_with_readme += 1  # using description as proxy

    total_repos = max(public_repos, 1)
    unique_languages = len(languages)

    # ---- Recent events for activity recency ----
    events = _get(f"/users/{username}/events/public?per_page=30") or []
    if events:
        fields_fetched += 1
    activity_recency = 9999
    if events and isinstance(events, list) and events:
        last_event_str = events[0].get("created_at", "") if isinstance(events[0], dict) else ""
        if last_event_str:
            try:
                last_event = datetime.fromisoformat(last_event_str.replace("Z", "+00:00"))
                activity_recency = (datetime.now(timezone.utc) - last_event).days
            except Exception:
                pass

    # ---- Commit regularity (from events) ----
    commit_days: Dict[str, int] = {}
    push_events = [e for e in events
                   if isinstance(e, dict) and e.get("type") == "PushEvent"]
    for ev in push_events:
        try:
            day = ev["created_at"][:10]
            commit_days[day] = commit_days.get(day, 0) + 1
        except Exception:
            pass

    commit_regularity = 0.0
    if len(commit_days) >= 3:
        import statistics
        try:
            commit_regularity = statistics.stdev(commit_days.values())
        except Exception:
            pass
    fields_fetched += 1

    # ---- Compute features ----
    features = {
        # Raw signals
        "followers": followers,
        "following": following,
        "public_repos": public_repos,
        "public_gists": public_gists,
        "account_age_days": account_age_days,
        "bio": bio,
        "avatar_url": avatar_url,
        # Derived signals
        "repo_to_follower_ratio": public_repos / max(followers, 1),
        "commit_regularity": commit_regularity,
        "repo_diversity_score": unique_languages / max(total_repos, 1),
        "star_received_total": stars_total,
        "fork_ratio": forks_count / max(total_repos, 1),
        "readme_presence_ratio": repos_with_readme / max(total_repos, 1),
        "activity_recency": activity_recency,
        "ff_ratio": followers / max(following, 1),
        "unique_languages": unique_languages,
    }

    completeness = fields_fetched / fields_total

    logger.info(
        "GitHub enrichment for '%s': completeness=%.2f followers=%d repos=%d age=%dd",
        username, completeness, followers, public_repos, account_age_days,
    )

    return {
        "platform": "github",
        "username": username,
        "name": name,
        "followers": followers,
        "following": following,
        "posts": None,
        "bio": bio,
        "is_verified": False,
        "has_avatar": bool(avatar_url),
        "avatar_url": avatar_url,
        "account_age_days": int(account_age_days) if account_age_days else None,
        "location": user.get("location"),
        "website": user.get("blog"),
        "completeness": min(1.0, completeness),
        "raw_data": {
            "login": user.get("login"),
            "followers": followers,
            "following": following,
            "public_repos": public_repos,
            "public_gists": public_gists,
            "bio": bio,
            "created_at": created_at_str,
            "avatar_url": avatar_url,
            "name": name,
            "location": user.get("location"),
            "blog": user.get("blog"),
            "company": user.get("company"),
        },
        "features": features,
        "data_completeness_score": min(1.0, completeness),
        "public_repos": public_repos,
        "public_gists": public_gists,
        "company": user.get("company"),
    }
