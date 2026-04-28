"""
data_generator.py — Synthetic training data generator for fake profile detection.

Provides generate_fake_profiles(), generate_legit_profiles(), and generate_dataset()
for all 10 supported platforms. Used by the /generate-data Flask route.

Each generator produces realistic labeled profiles with platform-specific patterns:
  - Fake: burst creation, extreme ratios, spammy bios, bot usernames
  - Legit: organic growth, natural language bios, human-like posting patterns
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ML.config import LABEL_COLUMN, FAKE_VALUE, LEGIT_VALUE

logger = logging.getLogger(__name__)

RNG = np.random.default_rng(42)

PLATFORMS = [
    "instagram", "facebook", "x", "linkedin", "github",
    "discord", "youtube", "tiktok", "reddit", "snapchat",
]

# ---------------------------------------------------------------------------
# Bio / username helpers
# ---------------------------------------------------------------------------

# Celebrity / public figure bio templates (very short or empty is NORMAL)
CELEBRITY_BIO_TEMPLATES = [
    "",  # empty bio — very common for celebrities
    "Official account",
    "CEO of {company}",
    "Founder & CEO | {company}",
    "Actor. Director. Producer.",
    "Professional athlete | {team}",
    "Recording artist",
    "Entrepreneur | Investor",
    "Author of {book}",
    "Public speaker | Activist",
    "Politician | {role}",
    "Journalist at {outlet}",
    "Co-founder of {company}",
    "Official account. For business: press@agency.com",
    "Grammy-winning artist",
    "Professional athlete | Brand ambassador",
    "World-renowned chef | Cookbook author",
    "CEO | Author | Speaker",
    "Oscar-winning director",
    "World #1 | Foundation: @AidFund",
    "Fashion icon | Creative Director",
    "Entrepreneur | Investor | Philanthropist",
]

# Real celebrities often use real name or simple handles
CELEBRITY_USERNAME_PATTERNS = [
    "therock", "cristiano", "kimkardashian", "kyliejenner", "leomessi",
    "selenagomez", "justinbieber", "taylorswift", "beyonce", "arianagrande",
    "neymarjr", "elonmusk", "billgates", "timc00k", "jeffbezos",
    "nasa", "natgeo", "nike", "apple", "instagram",
    "meta", "microsoft", "google", "amazon", "tesla",
    "fcbarcelona", "realmadrid", "mancity", "nba", "nfl",
    "bbc", "cnn", "nytimes", "theguardian", "reuters",
]

_LEGIT_BIOS = [
    "photographer | travel 📷", "software engineer @ Google",
    "mom of 3 | foodie", "fitness coach 💪 DM for training plans",
    "Musician. Coffee addict.", "Developer | Open source enthusiast",
    "Marketing professional | keynote speaker", "Teacher & lifelong learner",
    "Entrepreneur | Founder @startup", "Dog lover | Hiking | Books",
    "Data scientist | ML researcher", "Chef | Recipe creator",
    "Artist | Digital creator", "Blogger | Lifestyle & wellness",
    "Traveler | 50+ countries visited", "News journalist @NYT",
    "Gaming streamer | Twitch partner", "Fashion designer | PR welcome",
    "Medical student | future doctor", "Environmental activist",
    "Wildlife photographer based in Kenya", "Architect | sustainable design",
    "Retired engineer | amateur astronomer", "Dance teacher | choreographer",
    "Nurse practitioner | women's health advocate",
    "Podcaster | interviewing founders", "UX designer @Meta",
    "High school teacher | making STEM fun", "Yoga instructor | wellness coach",
    "Freelance illustrator | art prints available",
]

_FAKE_BIOS = [
    "", "", "", "", "",   # many fake profiles have empty bios
    "Click link below 💰", "earn $500/day 👇",
    "FREE BTC airdrop → link in bio", "investment advisor DM me",
    "crypto signals ✅ 99% win rate", "follow for follow back",
    "♻️ repost to win $1000", "NFT investor | crypto tips",
    "📈 passive income daily", "DM me to grow your account fast",
    "promo available 📩", "btc eth airdrop free coins",
    "make money online → bio link", "official giveaway account",
    "FREE gift cards click here", "earn while you sleep 💸",
    "crypto millionaire | teach you how", "airdrop $ETH daily",
]

_LEGIT_USERNAMES = [
    "john_doe", "sarah_smith", "mike_2023", "emma.j", "alex_travel",
    "dev_pete", "photo_by_anna", "chef_marco", "fit_jessica", "code_master",
    "creative_dan", "lisa_writes", "mark_adventure", "nina_art", "paul_runs",
    "tom_techie", "grace_beauty", "daniel_f", "amy_joy", "chris_builds",
    "rachel_reads", "james_hikes", "olivia_draws", "ben_codes", "mia_cooks",
    "luke_designs", "sophia_teaches", "noah_films", "ava_dances", "ethan_games",
]

_FAKE_USERNAMES = [
    "user8374629", "acc29471838", "follow_back_101", "free_follower_bot",
    "1234567890ab", "xXx_profit_xXx", "crypto_king_9999", "bot_account_7",
    "a8f7d3k9p2", "zz_promo_zz", "9182736450", "earn_fast_2024",
    "q7w8e9r0t1y2", "airdrop_winner99", "real_human_01234", "get_rich_88",
    "bot291847362", "autoliker_pro", "f4f_master_99", "insta_boost_247",
]

# Realistic bios for casual / borderline legit users (no bio, short bios, lazy bios)
_BORDERLINE_BIOS = [
    "", "", "", "",          # Empty bio is NORMAL for casual users
    "just here",
    "hi",
    "🙂",
    "living life",
    "student",
    "lurker",
    "here for the memes",
    "follow back",           # ambiguous — can be legit or fake
    "new here",
    "just joined",
    "trying this out",
    "mostly lurking",
    "idk what to put here",
    "exploring",
    "casual user",
    "not really active",
]

_LEGIT_HEADLINES = [
    "Senior Software Engineer @ Amazon | Python & ML",
    "Marketing Director | B2B SaaS | Speaker",
    "Data Science Lead | ex-Google | PhD Stanford",
    "Product Manager | Fintech | Building the future",
    "UX Researcher | Human-centered design advocate",
    "Sales Director | Revenue growth specialist",
    "Operations Manager | Process optimization",
    "Financial Analyst | CFA | Investment research",
    "HR Business Partner | People & culture",
    "Content Strategist | Brand storytelling",
]

_FAKE_HEADLINES = [
    "💰 Financial Freedom Coach | DM to learn how",
    "🚀 Crypto Expert | 10x your investment",
    "Work from home | earn $1000/day guaranteed",
    "MLM Business Owner | Join my team",
    "Network Marketing Professional | passive income",
    "Bitcoin Investor | Signals & Tips | DM me",
    "Online Business Mentor | I'll teach you",
    "Forex Trader | 95% win rate | Copy my trades",
]

_LEGIT_CHANNEL_NAMES = [
    "TechWithTim", "JennsKitchen", "TravelDiaries", "CodeNewbie",
    "FitLife365", "ArtByMaria", "DailyPhilosophy", "GamingWithAlex",
    "CookingBasics", "ScienceExplained",
]
_FAKE_CHANNEL_NAMES = [
    "FREE MONEY CHANNEL", "EASY RICHES NOW", "BITCOIN SIGNALS 100%",
    "GET RICH QUICK OFFICIAL", "AIRDROP ALERTS DAILY",
]


def _rng_sample(arr: list, n: int) -> list:
    """Sample n items from arr with replacement using module RNG."""
    indices = RNG.integers(0, len(arr), n)
    return [arr[i] for i in indices]


def _fake_timestamps(n_posts: int, account_age_days: int) -> str:
    """Generate uniform/burst fake posting timestamps (JSON string)."""
    if n_posts <= 0:
        return "[]"
    base = datetime.now(timezone.utc) - timedelta(days=account_age_days)
    # Bots post at exact hourly intervals (very regular)
    interval_hours = max(1, account_age_days * 24 // max(n_posts, 1))
    timestamps = []
    for i in range(min(n_posts, 50)):
        t = base + timedelta(hours=i * interval_hours)
        timestamps.append(t.isoformat())
    return json.dumps(timestamps)


def _legit_timestamps(n_posts: int, account_age_days: int) -> str:
    """Generate irregular human-like posting timestamps (JSON string)."""
    if n_posts <= 0:
        return "[]"
    base = datetime.now(timezone.utc) - timedelta(days=account_age_days)
    timestamps = []
    current = base
    for _ in range(min(n_posts, 50)):
        # Irregular gaps: 0.5 to 72 hours
        gap_hours = float(RNG.exponential(scale=12.0))
        gap_hours = max(0.5, min(gap_hours, 72.0))
        current += timedelta(hours=gap_hours)
        if current > datetime.now(timezone.utc):
            break
        timestamps.append(current.isoformat())
    return json.dumps(timestamps)


def _discord_created_at(account_age_days: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=account_age_days)
    return dt.isoformat()


def _discord_joined_at(created_at_str: str, gap_seconds: float) -> str:
    try:
        from dateutil import parser as dp
        created = dp.parse(created_at_str)
    except Exception:
        created = datetime.now(timezone.utc) - timedelta(days=1)
    return (created + timedelta(seconds=gap_seconds)).isoformat()


# ---------------------------------------------------------------------------
# Temporal feature columns (pre-computed for synthetic data)
# ---------------------------------------------------------------------------

def _fake_temporal_cols(n: int) -> dict:
    """Generate fake-profile temporal feature columns."""
    return {
        "burst_score": RNG.uniform(0.6, 0.95, n).round(3),
        "posting_regularity_score": RNG.uniform(0.85, 0.99, n).round(3),
        "time_of_day_entropy": RNG.uniform(0.1, 0.4, n).round(3),
        "active_hours_count": RNG.integers(1, 5, n),
        "follow_velocity": RNG.uniform(50, 300, n).round(2),
        "duplicate_content_ratio": RNG.uniform(0.7, 0.99, n).round(3),
        "sentiment_variance": RNG.uniform(0.01, 0.1, n).round(4),
        "profile_completeness_score": RNG.uniform(0.0, 0.3, n).round(3),
        "mutual_follower_ratio": RNG.uniform(0.0, 0.05, n).round(4),
    }


def _legit_temporal_cols(n: int) -> dict:
    """Generate legit-profile temporal feature columns."""
    return {
        "burst_score": RNG.uniform(0.0, 0.15, n).round(3),
        "posting_regularity_score": RNG.uniform(0.1, 0.55, n).round(3),
        "time_of_day_entropy": RNG.uniform(0.65, 0.99, n).round(3),
        "active_hours_count": RNG.integers(8, 19, n),
        "follow_velocity": RNG.uniform(0.1, 5, n).round(3),
        "duplicate_content_ratio": RNG.uniform(0.0, 0.2, n).round(3),
        "sentiment_variance": RNG.uniform(0.2, 0.7, n).round(4),
        "profile_completeness_score": RNG.uniform(0.6, 1.0, n).round(3),
        "mutual_follower_ratio": RNG.uniform(0.1, 0.6, n).round(4),
    }


# ---------------------------------------------------------------------------
# Platform-specific generators
# ---------------------------------------------------------------------------

def _gen_instagram_fake(n: int) -> List[dict]:
    rows = []
    n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)
    # Type 1: bots — high following, minimal followers, no posts
    for _ in range(n1):
        age = int(RNG.integers(1, 20))
        rows.append({
            "username": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "followers": int(RNG.integers(0, 50)),
            "following": int(RNG.integers(3000, 8000)),
            "posts": int(RNG.integers(0, 3)),
            "bio": _rng_sample(_FAKE_BIOS, 1)[0],
            "is_verified": 0,
            "post_timestamps_json": _fake_timestamps(int(RNG.integers(0, 3)), age),
        })
    # Type 2: spam — moderate followers, many posts, spammy bio
    for _ in range(n2):
        age = int(RNG.integers(5, 60))
        rows.append({
            "username": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "followers": int(RNG.integers(200, 5000)),
            "following": int(RNG.integers(2000, 7500)),
            "posts": int(RNG.integers(500, 5000)),
            "bio": _rng_sample(_FAKE_BIOS, 1)[0],
            "is_verified": 0,
            "post_timestamps_json": _fake_timestamps(50, age),
        })
    # Type 3: purchased followers — huge followers, no following/posts
    for _ in range(n3):
        age = int(RNG.integers(10, 90))
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": int(RNG.integers(10000, 500000)),
            "following": int(RNG.integers(0, 50)),
            "posts": int(RNG.integers(0, 10)),
            "bio": _rng_sample(_FAKE_BIOS, 1)[0],
            "is_verified": 0,
            "post_timestamps_json": _fake_timestamps(5, age),
        })
    return rows


def _gen_instagram_celebrity(n: int) -> List[dict]:
    """Generate celebrity-tier legitimate Instagram profiles (500K–500M followers)."""
    rows = []
    for _ in range(n):
        followers = int(np.clip(
            RNG.lognormal(np.log(5_000_000), 1.5),
            500_000, 500_000_000,
        ))
        following = int(RNG.integers(0, 2_001))       # celebrities follow very few people
        posts = int(np.clip(RNG.lognormal(np.log(1_000), 1.0), 50, 10_000))
        age = int(RNG.integers(1825, 7300))            # 5–20 year-old accounts
        bio_template = _rng_sample(CELEBRITY_BIO_TEMPLATES, 1)[0]
        rows.append({
            "username":    _rng_sample(CELEBRITY_USERNAME_PATTERNS, 1)[0],
            "followers":   followers,
            "following":   following,
            "posts":       posts,
            "bio":         bio_template,
            "is_verified": 1 if RNG.random() < 0.85 else 0,  # most but not all
            "account_age_days": age,
            "post_timestamps_json": _legit_timestamps(50, age),
        })
    return rows


def _gen_instagram_legit(n: int) -> List[dict]:
    """Generate legit Instagram profiles across casual/established/influencer tiers."""
    rows = []
    # 10% celebrity tier mixed in so the model sees this as legitimate
    n_celeb = max(1, int(n * 0.10))
    n_regular = n - n_celeb
    rows.extend(_gen_instagram_celebrity(n_celeb))

    for _ in range(n_regular):
        followers = int(np.clip(RNG.lognormal(6, 1.5), 100, 500_000))
        following = int(np.clip(followers * RNG.uniform(0.3, 2.0), 50, 5000))
        posts = int(np.clip(RNG.lognormal(4, 1), 15, 3000))
        age = int(RNG.integers(180, 2000))
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": followers,
            "following": following,
            "posts": posts,
            "bio": _rng_sample(_LEGIT_BIOS, 1)[0],
            "is_verified": 1 if RNG.random() < 0.05 else 0,
            "account_age_days": age,
            "post_timestamps_json": _legit_timestamps(50, age),
        })
    return rows


def _gen_instagram_borderline_legit(n: int) -> List[dict]:
    """Casual / new real users — low activity but no bot signals."""
    rows = []
    for _ in range(n):
        followers = int(RNG.integers(5, 500))       # Low but normal
        following = int(RNG.integers(50, 800))       # Following > followers is normal
        posts = int(RNG.integers(0, 50))             # Minimal activity
        age = int(RNG.integers(30, 730))             # Young accounts exist
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": followers,
            "following": following,
            "posts": posts,
            "bio": _rng_sample(_BORDERLINE_BIOS, 1)[0],
            "is_verified": 0,
            "post_timestamps_json": _legit_timestamps(posts, age),
        })
    return rows


def _gen_facebook_fake(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        rows.append({
            "name": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "friends": int(RNG.integers(0, 30)),
            "followers": int(RNG.integers(0, 100)),
            "posts": int(RNG.integers(0, 5)),
            "bio": _rng_sample(_FAKE_BIOS, 1)[0],
            "is_verified": 0,
        })
    return rows


def _gen_facebook_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        friends = int(np.clip(RNG.lognormal(5, 0.8), 50, 5000))
        rows.append({
            "name": _rng_sample(_LEGIT_USERNAMES, 1)[0].replace("_", " ").title(),
            "friends": friends,
            "followers": int(friends * RNG.uniform(0.5, 3.0)),
            "posts": int(np.clip(RNG.lognormal(3, 1), 10, 2000)),
            "bio": _rng_sample(_LEGIT_BIOS, 1)[0],
            "is_verified": 1 if RNG.random() < 0.03 else 0,
        })
    return rows


def _gen_x_fake(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        rows.append({
            "username": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "followers": int(RNG.integers(0, 80)),
            "following": int(RNG.integers(2000, 7500)),
            "tweets": int(RNG.integers(0, 5)),
            "bio": _rng_sample(_FAKE_BIOS, 1)[0],
            "is_verified": 0,
        })
    return rows


def _gen_x_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        followers = int(np.clip(RNG.lognormal(6, 1.5), 50, 200000))
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": followers,
            "following": int(np.clip(followers * RNG.uniform(0.1, 1.5), 10, 3000)),
            "tweets": int(np.clip(RNG.lognormal(5, 1), 50, 50000)),
            "bio": _rng_sample(_LEGIT_BIOS, 1)[0],
            "is_verified": 1 if RNG.random() < 0.04 else 0,
        })
    return rows


def _gen_linkedin_fake(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        rows.append({
            "name": _rng_sample(_FAKE_USERNAMES, 1)[0].replace("_", " ").title(),
            "connections": int(RNG.integers(0, 50)),
            "followers": int(RNG.integers(0, 100)),
            "headline": _rng_sample(_FAKE_HEADLINES, 1)[0],
            "about": _rng_sample(_FAKE_BIOS, 1)[0],
        })
    return rows


def _gen_linkedin_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        connections = int(np.clip(RNG.lognormal(5.5, 0.7), 50, 30000))
        rows.append({
            "name": _rng_sample(_LEGIT_USERNAMES, 1)[0].replace("_", " ").title(),
            "connections": connections,
            "followers": int(connections * RNG.uniform(0.5, 2.0)),
            "headline": _rng_sample(_LEGIT_HEADLINES, 1)[0],
            "about": _rng_sample(_LEGIT_BIOS, 1)[0],
        })
    return rows


def _gen_github_fake(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        # Fake accounts: very new, few repos, high following-to-follower ratio
        age_days = int(RNG.integers(1, 60))
        rows.append({
            "username": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "followers": int(np.clip(RNG.integers(0, 20), 0, 20)),
            "following": int(RNG.integers(200, 3000)),
            "public_repos": int(np.clip(RNG.integers(0, 3), 0, 3)),
            "public_gists": 0,
            "account_age_days": age_days,
            "bio": _rng_sample(_FAKE_BIOS, 1)[0],
        })
    return rows


def _gen_github_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        repos = int(np.clip(RNG.lognormal(2.5, 1.2), 1, 400))
        # Legit accounts: older, varied follower counts, some gists, real bios
        age_days = int(np.clip(RNG.lognormal(7.5, 1.0), 365, 5000))
        gists = int(np.clip(RNG.integers(0, 30), 0, 30))
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": int(np.clip(RNG.lognormal(3.5, 1.8), 1, 100000)),
            "following": int(np.clip(RNG.lognormal(3, 0.9), 1, 2000)),
            "public_repos": repos,
            "public_gists": gists,
            "account_age_days": age_days,
            "bio": _rng_sample(_LEGIT_BIOS, 1)[0],
        })
    return rows


def _gen_github_borderline_legit(n: int) -> List[dict]:
    """New / casual developers — few repos, short bio, but not bots."""
    rows = []
    for _ in range(n):
        age_days = int(RNG.integers(30, 400))        # Newer accounts
        repos = int(RNG.integers(0, 5))              # Just starting
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": int(RNG.integers(0, 30)),
            "following": int(RNG.integers(5, 200)),
            "public_repos": repos,
            "public_gists": 0,
            "account_age_days": age_days,
            "bio": _rng_sample(_BORDERLINE_BIOS, 1)[0],
        })
    return rows


def _gen_discord_fake(n: int) -> List[dict]:
    rows = []
    import time as _time
    for _ in range(n):
        age_days = int(RNG.integers(1, 7))
        created = _discord_created_at(age_days)
        gap_secs = float(RNG.uniform(0, 60))   # created and joined within 60 seconds
        rows.append({
            "timestamp_utc": created,
            "guild_id": str(int(RNG.integers(100000000, 999999999))),
            "guild_name": "Unknown Server",
            "user_id": str(int(RNG.integers(100000000000000000, 999999999999999999))),
            "username": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "created_at_utc": created,
            "joined_at_utc": _discord_joined_at(created, gap_secs),
            "account_age_days": age_days,
            "has_avatar": 0 if RNG.random() < 0.8 else 1,
            "suspicion_score": float(RNG.uniform(0.7, 1.0)),
            "reasons": "new_account,no_avatar",
            "action_taken": "flagged",
        })
    return rows


def _gen_discord_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        age_days = int(RNG.integers(200, 1500))
        created = _discord_created_at(age_days)
        gap_secs = float(RNG.uniform(3600, 86400 * 30))
        rows.append({
            "timestamp_utc": created,
            "guild_id": str(int(RNG.integers(100000000, 999999999))),
            "guild_name": "Community Server",
            "user_id": str(int(RNG.integers(100000000000000000, 999999999999999999))),
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "created_at_utc": created,
            "joined_at_utc": _discord_joined_at(created, gap_secs),
            "account_age_days": age_days,
            "has_avatar": 1,
            "suspicion_score": float(RNG.uniform(0.0, 0.2)),
            "reasons": "",
            "action_taken": "none",
        })
    return rows


def _gen_youtube_fake(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        videos = int(RNG.integers(200, 2000))
        rows.append({
            "channel_name": _rng_sample(_FAKE_CHANNEL_NAMES, 1)[0],
            "subscribers": int(RNG.integers(0, 500)),     # few subs despite many videos
            "videos": videos,
            "about": _rng_sample(_FAKE_BIOS, 1)[0],
        })
    return rows


def _gen_youtube_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        subs = int(np.clip(RNG.lognormal(8, 1.5), 100, 10000000))
        videos = int(np.clip(RNG.lognormal(3, 1), 5, 2000))
        rows.append({
            "channel_name": _rng_sample(_LEGIT_CHANNEL_NAMES, 1)[0],
            "subscribers": subs,
            "videos": videos,
            "about": _rng_sample(_LEGIT_BIOS, 1)[0],
        })
    return rows


def _gen_tiktok_fake(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        rows.append({
            "username": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "followers": int(RNG.integers(0, 100)),
            "following": int(RNG.integers(3000, 8000)),
            "videos": int(RNG.integers(0, 3)),
            "bio": _rng_sample(_FAKE_BIOS, 1)[0],
        })
    return rows


def _gen_tiktok_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        followers = int(np.clip(RNG.lognormal(7, 2), 100, 5000000))
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": followers,
            "following": int(np.clip(followers * RNG.uniform(0.01, 0.5), 10, 2000)),
            "videos": int(np.clip(RNG.lognormal(3, 1), 5, 1000)),
            "bio": _rng_sample(_LEGIT_BIOS, 1)[0],
        })
    return rows


def _gen_reddit_fake(n: int) -> List[dict]:
    rows = []
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for _ in range(n):
        age_days = int(RNG.integers(1, 30))
        created = (datetime.now(timezone.utc) - timedelta(days=age_days)).strftime("%Y-%m-%d")
        rows.append({
            "username": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "karma": int(RNG.integers(0, 50)),     # very low karma
            "created_at": created,
            "about": _rng_sample(_FAKE_BIOS, 1)[0],
        })
    return rows


def _gen_reddit_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        age_days = int(RNG.integers(365, 3000))
        created = (datetime.now(timezone.utc) - timedelta(days=age_days)).strftime("%Y-%m-%d")
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "karma": int(np.clip(RNG.lognormal(7, 1.5), 500, 500000)),
            "created_at": created,
            "about": _rng_sample(_LEGIT_BIOS, 1)[0],
        })
    return rows


def _gen_snapchat_fake(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        rows.append({
            "username": _rng_sample(_FAKE_USERNAMES, 1)[0],
            "score": int(RNG.integers(0, 50)),
            "bio": _rng_sample(_FAKE_BIOS, 1)[0],
        })
    return rows


def _gen_snapchat_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "score": int(np.clip(RNG.lognormal(8, 1.5), 500, 5000000)),
            "bio": _rng_sample(_LEGIT_BIOS, 1)[0],
        })
    return rows


def _gen_facebook_borderline_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        rows.append({
            "name": _rng_sample(_LEGIT_USERNAMES, 1)[0].replace("_", " ").title(),
            "friends": int(RNG.integers(0, 100)),
            "followers": int(RNG.integers(0, 200)),
            "posts": int(RNG.integers(0, 20)),
            "bio": _rng_sample(_BORDERLINE_BIOS, 1)[0],
            "is_verified": 0,
        })
    return rows


def _gen_x_borderline_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        followers = int(RNG.integers(5, 300))
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": followers,
            "following": int(RNG.integers(50, 600)),
            "tweets": int(RNG.integers(0, 100)),
            "bio": _rng_sample(_BORDERLINE_BIOS, 1)[0],
            "is_verified": 0,
        })
    return rows


def _gen_linkedin_borderline_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        connections = int(RNG.integers(5, 100))
        rows.append({
            "name": _rng_sample(_LEGIT_USERNAMES, 1)[0].replace("_", " ").title(),
            "connections": connections,
            "followers": int(connections * RNG.uniform(0.5, 2.0)),
            "headline": _rng_sample(_BORDERLINE_BIOS, 1)[0] or "Student",
            "about": _rng_sample(_BORDERLINE_BIOS, 1)[0],
        })
    return rows


def _gen_discord_borderline_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        age_days = int(RNG.integers(30, 300))
        created = _discord_created_at(age_days)
        gap_secs = float(RNG.uniform(3600, 86400 * 7))
        rows.append({
            "timestamp_utc": created,
            "guild_id": str(int(RNG.integers(100000000, 999999999))),
            "guild_name": "Community Server",
            "user_id": str(int(RNG.integers(100000000000000000, 999999999999999999))),
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "created_at_utc": created,
            "joined_at_utc": _discord_joined_at(created, gap_secs),
            "account_age_days": age_days,
            "has_avatar": int(RNG.random() > 0.4),  # Some don't have avatars
            "suspicion_score": float(RNG.uniform(0.1, 0.35)),
            "reasons": "",
            "action_taken": "none",
        })
    return rows


def _gen_youtube_borderline_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        subs = int(RNG.integers(0, 500))
        rows.append({
            "channel_name": _rng_sample(_LEGIT_CHANNEL_NAMES, 1)[0],
            "subscribers": subs,
            "videos": int(RNG.integers(0, 15)),
            "about": _rng_sample(_BORDERLINE_BIOS, 1)[0],
        })
    return rows


def _gen_tiktok_borderline_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        followers = int(RNG.integers(5, 300))
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "followers": followers,
            "following": int(RNG.integers(50, 600)),
            "videos": int(RNG.integers(0, 15)),
            "bio": _rng_sample(_BORDERLINE_BIOS, 1)[0],
        })
    return rows


def _gen_reddit_borderline_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        age_days = int(RNG.integers(30, 365))
        created = (datetime.now(timezone.utc) - timedelta(days=age_days)).strftime("%Y-%m-%d")
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "karma": int(RNG.integers(1, 500)),
            "created_at": created,
            "about": _rng_sample(_BORDERLINE_BIOS, 1)[0],
        })
    return rows


def _gen_snapchat_borderline_legit(n: int) -> List[dict]:
    rows = []
    for _ in range(n):
        rows.append({
            "username": _rng_sample(_LEGIT_USERNAMES, 1)[0],
            "score": int(RNG.integers(50, 3000)),
            "bio": _rng_sample(_BORDERLINE_BIOS, 1)[0],
        })
    return rows


# Celebrity generators dispatch table
# Platforms without a dedicated celebrity generator fall back to None (no supplement)
_CELEBRITY_GENERATORS: dict = {
    "instagram": _gen_instagram_celebrity,
    "facebook":  None,
    "x":         None,
    "linkedin":  None,
    "github":    None,
    "discord":   None,
    "youtube":   None,
    "tiktok":    None,
    "reddit":    None,
    "snapchat":  None,
}

# Dispatch tables
_FAKE_GENERATORS = {
    "instagram": _gen_instagram_fake,
    "facebook": _gen_facebook_fake,
    "x": _gen_x_fake,
    "linkedin": _gen_linkedin_fake,
    "github": _gen_github_fake,
    "discord": _gen_discord_fake,
    "youtube": _gen_youtube_fake,
    "tiktok": _gen_tiktok_fake,
    "reddit": _gen_reddit_fake,
    "snapchat": _gen_snapchat_fake,
}

_LEGIT_GENERATORS = {
    "instagram": _gen_instagram_legit,
    "facebook": _gen_facebook_legit,
    "x": _gen_x_legit,
    "linkedin": _gen_linkedin_legit,
    "github": _gen_github_legit,
    "discord": _gen_discord_legit,
    "youtube": _gen_youtube_legit,
    "tiktok": _gen_tiktok_legit,
    "reddit": _gen_reddit_legit,
    "snapchat": _gen_snapchat_legit,
}

# Borderline legit generators (casual real users that may look ambiguous)
_BORDERLINE_LEGIT_GENERATORS = {
    "instagram": _gen_instagram_borderline_legit,
    "facebook": _gen_facebook_borderline_legit,
    "x": _gen_x_borderline_legit,
    "linkedin": _gen_linkedin_borderline_legit,
    "github": _gen_github_borderline_legit,
    "discord": _gen_discord_borderline_legit,
    "youtube": _gen_youtube_borderline_legit,
    "tiktok": _gen_tiktok_borderline_legit,
    "reddit": _gen_reddit_borderline_legit,
    "snapchat": _gen_snapchat_borderline_legit,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_realistic_noise(df: pd.DataFrame, missing_rate: float = 0.15) -> pd.DataFrame:
    """
    Inject realistic missingness and value noise into a dataset.

    - Sets 10–30% of non-label numeric values to NaN (simulating incomplete data)
    - Adds ±5% Gaussian noise to numeric features
    - Occasionally swaps feature values between fake/legit rows (2-3% label noise)

    Args:
        df: DataFrame with label column.
        missing_rate: Fraction of cells to set missing (default 15%).

    Returns:
        Noisy copy of df.
    """
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in (LABEL_COLUMN, "is_fake", "fake", "target")]

    n_rows = len(df)

    # Add missingness to numeric columns
    for col in num_cols:
        if col in ("is_verified", "has_avatar", "has_profile_pic"):
            continue  # Don't nullify binary flags
        n_missing = max(0, int(n_rows * missing_rate * RNG.uniform(0.5, 1.5)))
        if n_missing > 0:
            missing_idx = RNG.choice(n_rows, min(n_missing, n_rows), replace=False)
            df.loc[missing_idx, col] = np.nan

    # Add Gaussian noise to remaining numeric values
    for col in num_cols:
        col_std = df[col].std()
        if col_std and col_std > 0:
            n_noisy = max(1, int(n_rows * 0.15))
            noisy_idx = RNG.choice(n_rows, n_noisy, replace=False)
            valid_mask = df.loc[noisy_idx, col].notna()
            valid_idx = [i for i in noisy_idx if not pd.isna(df.loc[i, col])]
            if valid_idx:
                noise = RNG.normal(0, col_std * 0.05, len(valid_idx))
                df.loc[valid_idx, col] = (df.loc[valid_idx, col] + noise).clip(lower=0)

    # Flip ~2% of labels (label noise prevents overfit)
    label_col = next(
        (c for c in (LABEL_COLUMN, "is_fake", "fake", "target") if c in df.columns), None
    )
    if label_col:
        n_flip = max(1, int(n_rows * 0.02))
        flip_idx = RNG.choice(n_rows, n_flip, replace=False)
        df.loc[flip_idx, label_col] = 1 - df.loc[flip_idx, label_col]

    return df


def generate_fake_profiles(platform: str, count: int) -> pd.DataFrame:
    """
    Generate `count` synthetic fake profiles for the given platform.

    Fake profiles exhibit bot-like characteristics:
    - Low account age, burst creation patterns
    - Extreme follower/following ratios
    - Spammy/crypto bios
    - Numeric/bot usernames
    - Pre-computed temporal behavioral columns

    Args:
        platform: One of the 10 supported platforms.
        count: Number of profiles to generate.

    Returns:
        DataFrame with `label=1` and platform-specific columns + temporal feature columns.
    """
    platform = platform.lower()
    if platform not in _FAKE_GENERATORS:
        raise ValueError(f"Unsupported platform: {platform}. Choose from {PLATFORMS}")
    if count < 1:
        raise ValueError("count must be >= 1")

    rows = _FAKE_GENERATORS[platform](count)
    df = pd.DataFrame(rows)
    df[LABEL_COLUMN] = FAKE_VALUE

    # Add temporal feature columns
    temporal_cols = _fake_temporal_cols(len(df))
    for col, vals in temporal_cols.items():
        df[col] = vals

    logger.info("Generated %d fake profiles for platform=%s", len(df), platform)
    return df


def generate_legit_profiles(platform: str, count: int) -> pd.DataFrame:
    """
    Generate `count` synthetic legitimate profiles for the given platform.

    Legit profiles exhibit organic behavior:
    - Natural account age distribution (months to years)
    - Organic follower/following growth
    - Varied, human-like bios and usernames
    - Irregular posting patterns

    Args:
        platform: One of the 10 supported platforms.
        count: Number of profiles to generate.

    Returns:
        DataFrame with `label=0` and platform-specific columns + temporal feature columns.
    """
    platform = platform.lower()
    if platform not in _LEGIT_GENERATORS:
        raise ValueError(f"Unsupported platform: {platform}. Choose from {PLATFORMS}")
    if count < 1:
        raise ValueError("count must be >= 1")

    rows = _LEGIT_GENERATORS[platform](count)
    df = pd.DataFrame(rows)
    df[LABEL_COLUMN] = LEGIT_VALUE

    # Add temporal feature columns
    temporal_cols = _legit_temporal_cols(len(df))
    for col, vals in temporal_cols.items():
        df[col] = vals

    logger.info("Generated %d legit profiles for platform=%s", len(df), platform)
    return df


def generate_dataset(
    platform: str,
    total_count: int = 1000,
    fake_ratio: float = 0.3,
    add_noise: bool = True,
    realistic_mode: bool = True,
    celebrity_ratio: float = 0.07,
) -> pd.DataFrame:
    """
    Generate a complete labeled dataset combining fake and legit profiles.

    When realistic_mode=True (default), uses empirically-realistic proportions:
      - 40% high-quality legit (established accounts, complete profiles)
      - 30% borderline legit (casual users, new accounts, sparse data)
      - 20% obvious fakes (bot patterns, spam)
      - 10% sophisticated fakes (semi-legit looking)

    celebrity_ratio: fraction of total that should be celebrity-tier legit profiles.
    These are generated in addition to the realistic-mode tiers and are critical for
    teaching the model that extreme follower ratios can be legitimate.

    This distribution teaches the model that sparse/incomplete profiles are
    NOT automatically fake — addressing the core confidence calibration problem.

    Args:
        platform: One of the 10 supported platforms.
        total_count: Total number of profiles (fake + legit).
        fake_ratio: Fraction of fake profiles (ignored when realistic_mode=True).
        add_noise: Whether to inject realistic missingness and noise.
        realistic_mode: Use 40/30/20/10 realistic distribution (recommended).

    Returns:
        Shuffled DataFrame with all platform columns, temporal feature columns, and `label`.
    """
    platform = platform.lower()
    if platform not in PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}")

    # Celebrity supplement — generated separately and always included
    n_celebrity = max(1, int(total_count * max(0.0, celebrity_ratio)))
    celebrity_gen = _CELEBRITY_GENERATORS.get(platform)
    if celebrity_gen and n_celebrity > 0:
        celeb_rows = celebrity_gen(n_celebrity)
        df_celebrity = pd.DataFrame(celeb_rows)
        df_celebrity[LABEL_COLUMN] = LEGIT_VALUE
        temporal_cols = _legit_temporal_cols(len(df_celebrity))
        for col, vals in temporal_cols.items():
            df_celebrity[col] = vals
    else:
        df_celebrity = None

    if realistic_mode:
        # Adjust remaining count to account for celebrity supplement
        remaining = max(10, total_count - (n_celebrity if celebrity_gen else 0))
        # 40% high-quality legit, 30% borderline legit, 20% obvious fakes, 10% sophisticated fakes
        n_hq_legit = max(1, int(remaining * 0.40))
        n_bl_legit = max(1, int(remaining * 0.30))
        n_fake_obv = max(1, int(remaining * 0.20))
        n_fake_soph = max(1, remaining - n_hq_legit - n_bl_legit - n_fake_obv)

        df_hq_legit = generate_legit_profiles(platform, n_hq_legit)

        # Borderline legit
        bl_gen = _BORDERLINE_LEGIT_GENERATORS.get(platform)
        if bl_gen:
            bl_rows = bl_gen(n_bl_legit)
            df_bl_legit = pd.DataFrame(bl_rows)
            df_bl_legit[LABEL_COLUMN] = LEGIT_VALUE
            temporal_cols = _legit_temporal_cols(len(df_bl_legit))
            for col, vals in temporal_cols.items():
                df_bl_legit[col] = vals
        else:
            df_bl_legit = generate_legit_profiles(platform, n_bl_legit)

        df_fake_obv = generate_fake_profiles(platform, n_fake_obv)
        df_fake_soph = generate_fake_profiles(platform, n_fake_soph)

        parts = [df_hq_legit, df_bl_legit, df_fake_obv, df_fake_soph]
        if df_celebrity is not None:
            parts.append(df_celebrity)
        df = pd.concat(parts, ignore_index=True)

        logger.info(
            "Realistic mode: platform=%s hq_legit=%d borderline=%d "
            "fake_obv=%d fake_soph=%d celebrity=%d",
            platform, n_hq_legit, n_bl_legit, n_fake_obv, n_fake_soph,
            len(df_celebrity) if df_celebrity is not None else 0,
        )
    else:
        # Legacy mode: simple fake/legit split + celebrity supplement
        fake_count = max(1, int(total_count * fake_ratio))
        legit_count = max(1, total_count - fake_count
                          - (n_celebrity if celebrity_gen else 0))
        df_fake = generate_fake_profiles(platform, fake_count)
        df_legit = generate_legit_profiles(platform, legit_count)
        parts = [df_fake, df_legit]
        if df_celebrity is not None:
            parts.append(df_celebrity)
        df = pd.concat(parts, ignore_index=True)

    if add_noise:
        df = inject_realistic_noise(df)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    actual_fake = int(df[LABEL_COLUMN].sum())
    actual_legit = len(df) - actual_fake
    logger.info(
        "Generated dataset: platform=%s total=%d fake=%d legit=%d fake_ratio=%.2f",
        platform, len(df), actual_fake, actual_legit, actual_fake / max(len(df), 1),
    )

    return df


def save_dataset(df: pd.DataFrame, platform: str, output_dir: str = "csv") -> Path:
    """
    Save the generated dataset to a CSV file.

    Args:
        df: DataFrame to save.
        platform: Platform name (used for filename).
        output_dir: Directory to save to (created if missing).

    Returns:
        Path to the saved CSV file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{platform}_generated.csv"
    df.to_csv(path, index=False)
    logger.info("Saved dataset to %s", path)
    return path


def get_generation_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics about a generated dataset.

    Args:
        df: Generated DataFrame with a `label` column.

    Returns:
        Dict with: total, fake_count, legit_count, fake_ratio, column_count.
    """
    total = len(df)
    fake_count = int(df[LABEL_COLUMN].sum()) if LABEL_COLUMN in df.columns else 0
    legit_count = total - fake_count
    return {
        "total": total,
        "fake_count": fake_count,
        "legit_count": legit_count,
        "fake_ratio": round(fake_count / max(total, 1), 3),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }
