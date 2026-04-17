"""
Generate realistic synthetic training datasets for all 10 platforms.

Each platform dataset includes FOUR profile tiers for legitimate accounts:
  - Celebrity   : millions of followers, verified, established content
  - Influencer  : 100k–1M followers, often verified
  - Regular     : organic growth, natural metrics
  - New user    : small but genuine accounts

And FOUR fake profile sub-types:
  - Bot          : aggressive following, no content, numeric username
  - Spam         : high post volume, spammy bio, promo links
  - Purchased    : bought followers, no engagement
  - Impersonator : looks legit but key signals off (e.g. unverified celeb clone)

Run: python generate_datasets.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path("csv")
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────────

def sample(n, low, high, skew="none"):
    if skew == "log":
        return np.clip(RNG.lognormal(np.log(max(low, 1)), 1.2, n).astype(int), low, high)
    if skew == "low":
        return np.clip((RNG.exponential(scale=(high - low) / 5, size=n) + low).astype(int), low, high)
    if skew == "high":
        return np.clip(high - RNG.exponential(scale=(high - low) / 5, size=n).astype(int), low, high)
    return RNG.integers(low, high + 1, n)


def rand_bio(n, mode="legit"):
    legit = [
        "photographer | travel 📷", "software engineer @ Google",
        "mom of 3 | foodie 🍕", "fitness coach 💪 DM for plans",
        "Musician. Coffee addict. ☕", "Developer | Open source enthusiast",
        "Marketing professional | keynote speaker", "Teacher & lifelong learner 📚",
        "Entrepreneur | Founder @startup", "Dog lover 🐾 | Hiking | Books",
        "Data scientist | ML researcher", "Chef | Recipe creator 🍳",
        "Artist | Digital creator ✨", "Blogger | Lifestyle & wellness",
        "Traveler | 50+ countries visited ✈️", "News journalist @NYT",
        "Gaming streamer | Twitch partner 🎮", "Fashion designer | PR welcome",
        "Medical student | future doctor 🩺", "Environmental activist 🌱",
        "Wildlife photographer based in Kenya", "Architect | sustainable design",
        "Retired engineer | amateur astronomer 🔭", "Dance teacher | choreographer",
        "Nurse practitioner | women's health advocate", "Podcaster | interviewing founders",
        "UX designer @Meta", "High school teacher | making STEM fun",
        "Yoga instructor | wellness coach 🧘", "Freelance illustrator | art prints available",
    ]
    celebrity_bio = [
        "Official account. For business: manager@agency.com",
        "Grammy-winning artist 🎵 | New album out now",
        "Hollywood actor | Upcoming: Summer Blockbuster 2025",
        "Professional athlete | Team captain | Brand ambassador",
        "World-renowned chef 👨‍🍳 | Author of 3 cookbooks",
        "Nobel Peace Prize laureate | Activist",
        "CEO @Fortune500 | Author | Speaker",
        "International supermodel | Agency: IMG Models",
        "Iconic musician 🎸 | 40+ years on stage",
        "Oscar-winning director | Production company: @StudioX",
        "World #1 tennis player 🎾 | Foundation: @AidFundation",
        "Fashion icon | Creative Director @LuxuryBrand",
    ]
    fake = [
        "", "", "", "", "",
        "Click link below 💰", "earn $500/day 👇",
        "FREE BTC airdrop → link in bio", "investment advisor DM me",
        "crypto signals ✅ 99% win rate", "follow for follow back",
        "♻️ repost to win $1000", "NFT investor | crypto tips",
        "📈 passive income daily", "DM me to grow your account fast",
        "promo available 📩", "btc eth airdrop free coins",
        "🔞 link in bio", "Get free followers — click bio link",
        "Real estate investor | DM for passive income secrets",
    ]
    if mode == "legit":
        return RNG.choice(legit, n)
    if mode == "celebrity":
        return RNG.choice(celebrity_bio, n)
    return RNG.choice(fake, n)


def rand_username(n, mode="legit"):
    legit = [
        "john_doe", "sarah_smith", "mike_2023", "emma.j", "alex_travel",
        "dev_pete", "photo_by_anna", "chef_marco", "fit_jessica", "code_master",
        "creative_dan", "lisa_writes", "mark_adventure", "nina_art", "paul_runs",
        "tom_techie", "grace_beauty", "daniel_f", "amy_joy", "chris_builds",
        "the_real_sam", "everyday_elena", "journeys_with_kai", "b.scott.official",
        "mindful_maya", "lena_captures", "surf_with_evan", "botanist_beth",
    ]
    celebrity = [
        "therock", "kimkardashian", "cristiano", "arianagrande", "selenagomez",
        "kyliejenner", "justinbieber", "taylorswift", "beyonce", "neymarjr",
        "elonmusk", "oprah", "billgates", "jeffbezos", "tim_cook",
        "nasa", "natgeo", "nike", "realmadrid", "fcbarcelona",
    ]
    fake = [
        "user8374629", "acc29471838", "follow_back_101", "free_follower_bot",
        "1234567890ab", "xXx_profit_xXx", "crypto_king_9999", "bot_account_7",
        "a8f7d3k9p2", "zz_promo_zz", "9182736450", "earn_fast_2024",
        "q7w8e9r0t1y2", "airdrop_winner99", "real_human_01234", "get_rich_88",
        "free_followers_now", "99882736450", "tradebot_fx", "nft_dropper_69",
    ]
    if mode == "celebrity":
        return RNG.choice(celebrity, n)
    if mode == "legit":
        return RNG.choice(legit, n)
    return RNG.choice(fake, n)


# ── 1. Instagram ─────────────────────────────────────────────────────────────

def gen_instagram(n_fake=1200, n_legit=1200):
    rows = []

    # ── Legit tiers ──────────────────────────────────────────────────────────
    # Celebrity (verified, 1M+)
    n_celeb = n_legit // 5
    for i in range(n_celeb):
        f = int(RNG.integers(1_000_000, 300_000_000))
        rows.append({
            "username":       rand_username(1, "celebrity")[0],
            "followers":      f,
            "following":      int(RNG.integers(100, 2000)),
            "posts":          int(RNG.integers(300, 5000)),
            "bio":            rand_bio(1, "celebrity")[0],
            "is_verified":    1,
            "has_profile_pic": 1,
            "label": 0,
        })

    # Influencer (100k–1M, sometimes verified)
    n_infl = n_legit // 5
    for i in range(n_infl):
        f = int(RNG.integers(100_000, 1_000_000))
        rows.append({
            "username":       rand_username(1, "legit")[0],
            "followers":      f,
            "following":      int(RNG.integers(200, 3000)),
            "posts":          int(RNG.integers(100, 3000)),
            "bio":            rand_bio(1, "legit")[0],
            "is_verified":    int(RNG.random() < 0.3),
            "has_profile_pic": 1,
            "label": 0,
        })

    # Regular users
    n_regular = n_legit - n_celeb - n_infl
    # Split regular into established (60%) and borderline/casual (40%)
    n_estab = int(n_regular * 0.60)
    n_border = n_regular - n_estab
    followers = sample(n_estab, 50, 50_000, "log")
    following = np.clip((followers * RNG.uniform(0.3, 2.5, n_estab)).astype(int), 20, 5000)
    posts = sample(n_estab, 5, 2000, "log")
    for i in range(n_estab):
        rows.append({
            "username":       rand_username(1, "legit")[0],
            "followers":      int(followers[i]),
            "following":      int(following[i]),
            "posts":          int(posts[i]),
            "bio":            rand_bio(1, "legit")[0],
            "is_verified":    int(RNG.random() < 0.01),
            "has_profile_pic": 1,
            "label": 0,
        })
    # Borderline legit: casual / new users (low metrics but no bot signals)
    _borderline_bios = ["", "", "just here", "hi", "🙂", "living life", "student",
                        "lurker", "here for the memes", "new here", "trying this out",
                        "mostly lurking", "casual user", "not really active", "exploring"]
    for _ in range(n_border):
        f = int(RNG.integers(5, 500))
        rows.append({
            "username":       rand_username(1, "legit")[0],
            "followers":      f,
            "following":      int(RNG.integers(50, 800)),
            "posts":          int(RNG.integers(0, 50)),
            "bio":            RNG.choice(_borderline_bios),
            "is_verified":    0,
            "has_profile_pic": int(RNG.random() > 0.2),
            "label": 0,
        })

    # ── Fake tiers ───────────────────────────────────────────────────────────
    q = n_fake // 4

    # Bot (aggressive following, no content)
    for i in range(q):
        rows.append({
            "username":       rand_username(1, "fake")[0],
            "followers":      int(RNG.integers(0, 80)),
            "following":      int(RNG.integers(3000, 7500)),
            "posts":          int(RNG.integers(0, 5)),
            "bio":            rand_bio(1, "fake")[0],
            "is_verified":    0,
            "has_profile_pic": int(RNG.random() > 0.6),
            "label": 1,
        })

    # Spam (high post volume, spammy bio)
    for i in range(q):
        rows.append({
            "username":       rand_username(1, "fake")[0],
            "followers":      int(RNG.integers(200, 8000)),
            "following":      int(RNG.integers(2000, 7500)),
            "posts":          int(RNG.integers(1000, 8000)),
            "bio":            rand_bio(1, "fake")[0],
            "is_verified":    0,
            "has_profile_pic": 1,
            "label": 1,
        })

    # Purchased followers (many followers, near-zero engagement, no verification)
    for i in range(q):
        rows.append({
            "username":       rand_username(1, "legit")[0],  # plausible username
            "followers":      int(RNG.integers(10_000, 500_000)),
            "following":      int(RNG.integers(0, 100)),
            "posts":          int(RNG.integers(0, 15)),
            "bio":            rand_bio(1, "fake")[0],
            "is_verified":    0,
            "has_profile_pic": int(RNG.random() > 0.4),
            "label": 1,
        })

    # Impersonator (looks legit but unverified, low age signals)
    for i in range(n_fake - 3 * q):
        f = int(RNG.integers(50_000, 2_000_000))  # high followers but no verify
        rows.append({
            "username":       rand_username(1, "celebrity")[0] + str(int(RNG.integers(1, 999))),
            "followers":      f,
            "following":      int(RNG.integers(0, 50)),
            "posts":          int(RNG.integers(5, 50)),
            "bio":            "Official fan account",  # impersonator signal
            "is_verified":    0,  # not verified — this is the key differentiator
            "has_profile_pic": 1,
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "instagram_train.csv", index=False)
    print(f"Instagram: {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 2. Facebook ──────────────────────────────────────────────────────────────

def gen_facebook(n_fake=1000, n_legit=1000):
    rows = []
    names_legit = ["John Smith", "Sarah Jones", "Mike Brown", "Emma Davis",
                   "Anna Lee", "Chris Martin", "Lisa White", "David Clark",
                   "Priya Sharma", "Ahmed Hassan", "Maria Garcia", "Yuki Tanaka"]
    names_fake  = ["User 8374", "Account 293", "Person 1928", "Free Follower",
                   "Profit King", "Crypto Bob", "Earn Fast", "Real Human 001",
                   "Investment Guru", "Money Maker", "Click Here Now", "Bot 9999"]

    # Celebrity tier
    for _ in range(n_legit // 5):
        rows.append({
            "name": RNG.choice(["The Rock", "Cristiano Ronaldo", "Selena Gomez",
                                "Taylor Swift", "Kim Kardashian", "Ariana Grande"]),
            "friends":    0,  # public pages don't have friends
            "followers":  int(RNG.integers(10_000_000, 200_000_000)),
            "posts":      int(RNG.integers(500, 5000)),
            "bio":        rand_bio(1, "celebrity")[0],
            "is_verified": 1,
            "has_profile_pic": 1,
            "label": 0,
        })

    # Regular legit
    for _ in range(n_legit - n_legit // 5):
        f = int(sample(1, 50, 15_000, "log")[0])
        rows.append({
            "name":       RNG.choice(names_legit),
            "friends":    int(RNG.integers(50, 2000)),
            "followers":  f,
            "posts":      int(RNG.integers(10, 1000)),
            "bio":        rand_bio(1, "legit")[0],
            "is_verified": int(RNG.random() < 0.02),
            "has_profile_pic": 1,
            "label": 0,
        })

    # Fake
    for _ in range(n_fake):
        rows.append({
            "name":       RNG.choice(names_fake),
            "friends":    int(RNG.integers(0, 30)),
            "followers":  int(RNG.integers(0, 200)),
            "posts":      int(RNG.integers(0, 8)),
            "bio":        rand_bio(1, "fake")[0],
            "is_verified": 0,
            "has_profile_pic": int(RNG.random() > 0.6),
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "facebook_train.csv", index=False)
    print(f"Facebook:  {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 3. X / Twitter ───────────────────────────────────────────────────────────

def gen_x(n_fake=1400, n_legit=1400):
    rows = []

    # Celebrity
    for _ in range(n_legit // 5):
        f = int(RNG.integers(5_000_000, 200_000_000))
        rows.append({
            "username":      rand_username(1, "celebrity")[0],
            "followers":     f,
            "following":     int(RNG.integers(100, 2000)),
            "tweets":        int(RNG.integers(5000, 100_000)),
            "bio":           rand_bio(1, "celebrity")[0],
            "is_verified":   1,
            "listed_count":  int(RNG.integers(10_000, 500_000)),
            "has_profile_pic": 1,
            "label": 0,
        })

    # Influencer
    for _ in range(n_legit // 5):
        f = int(RNG.integers(100_000, 5_000_000))
        rows.append({
            "username":     rand_username(1, "legit")[0],
            "followers":    f,
            "following":    int(RNG.integers(200, 5000)),
            "tweets":       int(RNG.integers(500, 50_000)),
            "bio":          rand_bio(1, "legit")[0],
            "is_verified":  int(RNG.random() < 0.3),
            "listed_count": int(RNG.integers(500, 50_000)),
            "has_profile_pic": 1,
            "label": 0,
        })

    # Regular
    for _ in range(n_legit - 2 * (n_legit // 5)):
        f = int(sample(1, 50, 100_000, "log")[0])
        rows.append({
            "username":     rand_username(1, "legit")[0],
            "followers":    f,
            "following":    int(np.clip(f * RNG.uniform(0.2, 2.5), 10, 5000)),
            "tweets":       int(sample(1, 100, 50_000, "log")[0]),
            "bio":          rand_bio(1, "legit")[0],
            "is_verified":  int(RNG.random() < 0.02),
            "listed_count": int(RNG.integers(0, 1000)),
            "has_profile_pic": 1,
            "label": 0,
        })

    q = n_fake // 4
    # Bot: aggressive following
    for _ in range(q):
        rows.append({
            "username":     rand_username(1, "fake")[0],
            "followers":    int(RNG.integers(0, 100)),
            "following":    int(RNG.integers(3000, 5000)),
            "tweets":       int(RNG.integers(0, 50)),
            "bio":          rand_bio(1, "fake")[0],
            "is_verified":  0,
            "listed_count": 0,
            "has_profile_pic": int(RNG.random() > 0.5),
            "label": 1,
        })

    # Spam: high tweet volume
    for _ in range(q):
        rows.append({
            "username":     rand_username(1, "fake")[0],
            "followers":    int(RNG.integers(100, 5000)),
            "following":    int(RNG.integers(3000, 5000)),
            "tweets":       int(RNG.integers(50_000, 500_000)),
            "bio":          rand_bio(1, "fake")[0],
            "is_verified":  0,
            "listed_count": 0,
            "has_profile_pic": 1,
            "label": 1,
        })

    # Purchased follower: high followers, low activity
    for _ in range(q):
        rows.append({
            "username":     rand_username(1, "legit")[0],
            "followers":    int(RNG.integers(20_000, 500_000)),
            "following":    int(RNG.integers(0, 50)),
            "tweets":       int(RNG.integers(0, 20)),
            "bio":          rand_bio(1, "fake")[0],
            "is_verified":  0,
            "listed_count": 0,
            "has_profile_pic": int(RNG.random() > 0.5),
            "label": 1,
        })

    # Impersonator
    for _ in range(n_fake - 3 * q):
        rows.append({
            "username":     rand_username(1, "celebrity")[0] + str(int(RNG.integers(1, 999))),
            "followers":    int(RNG.integers(1000, 500_000)),
            "following":    int(RNG.integers(0, 100)),
            "tweets":       int(RNG.integers(10, 200)),
            "bio":          "Fan account. Not official.",
            "is_verified":  0,
            "listed_count": 0,
            "has_profile_pic": 1,
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "x_train.csv", index=False)
    print(f"X/Twitter: {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 4. LinkedIn ───────────────────────────────────────────────────────────────

def gen_linkedin(n_fake=1000, n_legit=1000):
    legit_headlines = [
        "Software Engineer at Google | Python | ML",
        "Marketing Manager at Unilever | Brand Strategy",
        "Data Scientist | NLP | Deep Learning",
        "Product Manager | Agile | SaaS",
        "Financial Analyst | CFA Level 3",
        "UX Designer | Figma | User Research",
        "Business Development | Startups | VC",
        "HR Manager | Talent Acquisition",
        "Civil Engineer | Infrastructure Projects",
        "PhD Candidate | AI Ethics",
        "Sales Director | Enterprise Software",
        "Consultant | McKinsey & Company",
        "Journalist | The New York Times",
        "Professor | Harvard University | Economics",
    ]
    celeb_headlines = [
        "CEO | Fortune 500 | Author | Speaker",
        "Founder & CEO | Unicorn Startup | Forbes 30 Under 30",
        "Nobel Laureate | Professor @Stanford",
        "Former President | Advocate | Author",
    ]
    fake_headlines = [
        "Investment Advisor | Earn 200% Returns",
        "CEO of multiple companies | DM for collab",
        "Forex Trader | 99% win rate",
        "Crypto Expert | Free signals",
        "Network Marketing | Join my team",
        "Life Coach | Unlocking your potential",
        "Real Estate Investor | Passive Income",
        "MLM Specialist | 6-figure income",
        "", "Self-employed | Ask me how",
    ]

    rows = []
    names = ["John Smith", "Sarah Jones", "Mike Brown", "Emma Davis",
             "Anna Lee", "Chris Martin", "Lisa White", "David Clark",
             "Priya Sharma", "Ahmed Hassan"]

    # Celebrity
    for _ in range(n_legit // 8):
        conn = int(RNG.integers(500, 30_000))
        rows.append({
            "name":         RNG.choice(["Elon Musk", "Satya Nadella", "Sundar Pichai",
                                        "Jensen Huang", "Tim Cook", "Sheryl Sandberg"]),
            "connections":  conn,
            "followers":    int(RNG.integers(500_000, 10_000_000)),
            "headline":     RNG.choice(celeb_headlines),
            "about":        "Leader, entrepreneur, and visionary. Shaping the future of technology.",
            "endorsements": int(RNG.integers(500, 5000)),
            "label": 0,
        })

    # Regular legit
    for _ in range(n_legit - n_legit // 8):
        conn = int(RNG.integers(50, 500))
        rows.append({
            "name":         RNG.choice(names),
            "connections":  conn,
            "followers":    int(conn * RNG.uniform(0.8, 3.0)),
            "headline":     RNG.choice(legit_headlines),
            "about":        f"Professional with {int(RNG.integers(2, 20))} years experience.",
            "endorsements": int(RNG.integers(5, 300)),
            "label": 0,
        })

    # Fake
    for _ in range(n_fake):
        conn = int(RNG.integers(0, 50))
        rows.append({
            "name":         RNG.choice(["User 8374", "Profit Guru", "Earn Fast Now",
                                        "Crypto King", "Investment Pro", "MLM Leader"]),
            "connections":  conn,
            "followers":    int(RNG.integers(0, 200)),
            "headline":     RNG.choice(fake_headlines),
            "about":        RNG.choice(["", "DM for business opportunity",
                                        "Helping people earn online", "Join my team"]),
            "endorsements": 0,
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "linkedin_train.csv", index=False)
    print(f"LinkedIn:  {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 5. GitHub ─────────────────────────────────────────────────────────────────

def gen_github(n_fake=1000, n_legit=1200):
    legit_bios = [
        "Open source contributor | Python enthusiast",
        "Full-stack developer | React | Node.js",
        "ML researcher | PhD student @MIT",
        "Building dev tools | Rust | Go",
        "DevOps engineer | Kubernetes | AWS",
        "Security researcher | CTF player",
        "Frontend developer | Vue | TypeScript",
        "Data engineer | Apache Spark",
        "Game developer | Unity | C#",
        "Embedded systems | IoT | C++",
        "Founder of @cool-oss-project",
        "Passionate about open science and reproducibility",
    ]
    celeb_bios = [
        "Creator of Linux | Software engineer",
        "Founder @GitHub | Developer tools",
        "Creator of Python | Developer at heart",
        "Co-founder @stripe | Building developer tools",
    ]
    fake_bios = ["", "", "", "Follow for follow", "crypto investor", "NFT artist | DM me"]

    rows = []

    # Celebrity (Linus-level)
    for _ in range(n_legit // 6):
        f = int(RNG.integers(50_000, 500_000))
        rows.append({
            "username":     rand_username(1, "celebrity")[0],
            "followers":    f,
            "following":    int(RNG.integers(0, 500)),
            "public_repos": int(RNG.integers(10, 200)),
            "bio":          RNG.choice(celeb_bios),
            "public_gists": int(RNG.integers(0, 100)),
            "label": 0,
        })

    # Regular legit
    for _ in range(n_legit - n_legit // 6):
        f = int(sample(1, 5, 5000, "log")[0])
        rows.append({
            "username":     rand_username(1, "legit")[0],
            "followers":    f,
            "following":    int(np.clip(f * RNG.uniform(0.1, 3.0), 0, 1000)),
            "public_repos": int(RNG.integers(3, 200)),
            "bio":          RNG.choice(legit_bios),
            "public_gists": int(RNG.integers(0, 50)),
            "label": 0,
        })

    # Fake: star-farmers, spammers, low activity
    for _ in range(n_fake):
        rows.append({
            "username":     rand_username(1, "fake")[0],
            "followers":    int(RNG.integers(0, 20)),
            "following":    int(RNG.integers(500, 5000)),
            "public_repos": int(RNG.integers(0, 3)),
            "bio":          RNG.choice(fake_bios),
            "public_gists": 0,
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "github_train.csv", index=False)
    print(f"GitHub:    {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 6. Discord ────────────────────────────────────────────────────────────────

def gen_discord(n_fake=1000, n_legit=1000):
    from datetime import datetime, timedelta

    rows = []
    base_ts = datetime(2024, 6, 1)

    # Legit
    for _ in range(n_legit):
        age_days = int(RNG.integers(90, 2500))
        created  = base_ts - timedelta(days=age_days)
        joined_gap = int(RNG.integers(1, min(age_days // 2, 500)))
        rows.append({
            "timestamp_utc":  base_ts.isoformat(),
            "guild_id":       "1234567890",
            "guild_name":     "Community Hub",
            "user_id":        str(RNG.integers(100_000_000_000, 999_999_999_999)),
            "username":       rand_username(1, "legit")[0],
            "created_at_utc": created.isoformat(),
            "joined_at_utc":  (created + timedelta(days=joined_gap)).isoformat(),
            "account_age_days": age_days,
            "has_avatar":     1,
            "suspicion_score": float(RNG.uniform(0, 0.25)),
            "reasons":        "",
            "action_taken":   "none",
            "label": 0,
        })

    # Fake (raid bots, new accounts)
    q = n_fake // 3
    for _ in range(q):  # raid bots: new + join immediately
        age_days = int(RNG.integers(0, 7))
        created  = base_ts - timedelta(days=age_days)
        rows.append({
            "timestamp_utc":  base_ts.isoformat(),
            "guild_id":       "1234567890",
            "guild_name":     "Community Hub",
            "user_id":        str(RNG.integers(100_000_000_000, 999_999_999_999)),
            "username":       rand_username(1, "fake")[0],
            "created_at_utc": created.isoformat(),
            "joined_at_utc":  created.isoformat(),
            "account_age_days": age_days,
            "has_avatar":     0,
            "suspicion_score": float(RNG.uniform(0.7, 1.0)),
            "reasons":        "new_account,no_avatar,rapid_join",
            "action_taken":   "flagged",
            "label": 1,
        })

    for _ in range(q):  # spam bots: slightly older, no avatar
        age_days = int(RNG.integers(7, 45))
        created  = base_ts - timedelta(days=age_days)
        rows.append({
            "timestamp_utc":  base_ts.isoformat(),
            "guild_id":       "1234567890",
            "guild_name":     "Community Hub",
            "user_id":        str(RNG.integers(100_000_000_000, 999_999_999_999)),
            "username":       rand_username(1, "fake")[0],
            "created_at_utc": created.isoformat(),
            "joined_at_utc":  (created + timedelta(hours=int(RNG.integers(0, 48)))).isoformat(),
            "account_age_days": age_days,
            "has_avatar":     0,
            "suspicion_score": float(RNG.uniform(0.5, 0.9)),
            "reasons":        "suspicious_username,no_avatar",
            "action_taken":   "warned",
            "label": 1,
        })

    for _ in range(n_fake - 2 * q):  # dormant bots: older but pattern off
        age_days = int(RNG.integers(30, 180))
        created  = base_ts - timedelta(days=age_days)
        rows.append({
            "timestamp_utc":  base_ts.isoformat(),
            "guild_id":       "1234567890",
            "guild_name":     "Community Hub",
            "user_id":        str(RNG.integers(100_000_000_000, 999_999_999_999)),
            "username":       rand_username(1, "fake")[0],
            "created_at_utc": created.isoformat(),
            "joined_at_utc":  created.isoformat(),
            "account_age_days": age_days,
            "has_avatar":     0,
            "suspicion_score": float(RNG.uniform(0.55, 0.85)),
            "reasons":        "dormant_bot,pattern_mismatch",
            "action_taken":   "flagged",
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "discord_train.csv", index=False)
    print(f"Discord:   {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 7. YouTube ────────────────────────────────────────────────────────────────

def gen_youtube(n_fake=1000, n_legit=1000):
    legit_abouts = [
        "Gaming channel | daily uploads", "Travel vlogs from around the world",
        "Cooking tutorials and recipes 🍳", "Tech reviews and unboxings",
        "Fitness and workout videos 💪", "Educational content for students",
        "Music covers and originals 🎵", "DIY and home projects",
        "News and current affairs", "Comedy sketches and skits",
        "Science experiments for kids 🔬", "True crime documentary channel",
    ]
    celeb_abouts = [
        "Official YouTube channel of MrBeast. Subscribe!",
        "PewDiePie — gaming, vlogs, memes.",
        "T-Series — Bollywood music and film",
        "Official channel of NASA",
    ]
    fake_abouts = [
        "", "", "Subscribe for free giveaways",
        "Make money online — click here", "Crypto investment tips",
        "SUBSCRIBE NOW for $100 giveaway", "Free gift cards daily",
        "Earn $500/day from home", "NFT and crypto alpha",
    ]

    rows = []

    # Celebrity
    for _ in range(n_legit // 5):
        subs = int(RNG.integers(10_000_000, 250_000_000))
        rows.append({
            "channel_name":  RNG.choice(["MrBeast", "PewDiePie", "T-Series",
                                          "NASA", "TED", "NatGeo", "BBC"]),
            "subscribers":   subs,
            "videos":        int(RNG.integers(200, 5000)),
            "about":         RNG.choice(celeb_abouts),
            "avg_views":     int(subs * RNG.uniform(0.05, 0.3)),
            "is_verified":   1,
            "label": 0,
        })

    # Regular legit
    for _ in range(n_legit - n_legit // 5):
        subs = int(sample(1, 100, 2_000_000, "log")[0])
        rows.append({
            "channel_name":  RNG.choice(["TechWithMike", "CookingWithSarah",
                                          "GamingPro", "FitnessFirst", "MusicMaestro"]),
            "subscribers":   subs,
            "videos":        int(RNG.integers(10, 1000)),
            "about":         RNG.choice(legit_abouts),
            "avg_views":     int(subs * RNG.uniform(0.005, 0.3)),
            "is_verified":   int(subs > 100_000 and RNG.random() < 0.3),
            "label": 0,
        })

    # Fake
    for _ in range(n_fake):
        rows.append({
            "channel_name":  RNG.choice(["Free Money Channel", "Crypto Tips Daily",
                                          "Easy Profit 2024", "BotChannel9999"]),
            "subscribers":   int(RNG.integers(0, 1000)),
            "videos":        int(RNG.integers(0, 5)),
            "about":         RNG.choice(fake_abouts),
            "avg_views":     int(RNG.integers(0, 50)),
            "is_verified":   0,
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "youtube_train.csv", index=False)
    print(f"YouTube:   {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 8. TikTok ─────────────────────────────────────────────────────────────────

def gen_tiktok(n_fake=1200, n_legit=1200):
    rows = []

    # Celebrity
    for _ in range(n_legit // 5):
        f = int(RNG.integers(5_000_000, 150_000_000))
        rows.append({
            "username":   rand_username(1, "celebrity")[0],
            "followers":  f,
            "following":  int(RNG.integers(0, 500)),
            "videos":     int(RNG.integers(100, 5000)),
            "bio":        rand_bio(1, "celebrity")[0],
            "likes":      int(f * RNG.uniform(10, 100)),
            "is_verified": 1,
            "label": 0,
        })

    # Regular legit
    for _ in range(n_legit - n_legit // 5):
        f = int(sample(1, 200, 2_000_000, "log")[0])
        rows.append({
            "username":   rand_username(1, "legit")[0],
            "followers":  f,
            "following":  int(np.clip(f * RNG.uniform(0.01, 0.5), 5, 5000)),
            "videos":     int(RNG.integers(5, 500)),
            "bio":        rand_bio(1, "legit")[0],
            "likes":      int(f * RNG.uniform(5, 50)),
            "is_verified": int(f > 1_000_000 and RNG.random() < 0.4),
            "label": 0,
        })

    # Fake
    q = n_fake // 3
    for _ in range(q):  # bot
        rows.append({
            "username":   rand_username(1, "fake")[0],
            "followers":  int(RNG.integers(0, 200)),
            "following":  int(RNG.integers(3000, 10_000)),
            "videos":     int(RNG.integers(0, 5)),
            "bio":        rand_bio(1, "fake")[0],
            "likes":      int(RNG.integers(0, 100)),
            "is_verified": 0,
            "label": 1,
        })
    for _ in range(q):  # spam
        rows.append({
            "username":   rand_username(1, "fake")[0],
            "followers":  int(RNG.integers(500, 10_000)),
            "following":  int(RNG.integers(5000, 10_000)),
            "videos":     int(RNG.integers(500, 5000)),
            "bio":        rand_bio(1, "fake")[0],
            "likes":      int(RNG.integers(0, 500)),
            "is_verified": 0,
            "label": 1,
        })
    for _ in range(n_fake - 2 * q):  # purchased
        rows.append({
            "username":   rand_username(1, "legit")[0],
            "followers":  int(RNG.integers(10_000, 500_000)),
            "following":  int(RNG.integers(0, 50)),
            "videos":     int(RNG.integers(0, 10)),
            "bio":        rand_bio(1, "fake")[0],
            "likes":      int(RNG.integers(0, 1000)),
            "is_verified": 0,
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "tiktok_train.csv", index=False)
    print(f"TikTok:    {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 9. Reddit ─────────────────────────────────────────────────────────────────

def gen_reddit(n_fake=1200, n_legit=1200):
    from datetime import datetime, timedelta

    base = datetime(2024, 6, 1)
    legit_abouts = [
        "Regular reddit user. Gaming and tech.", "I post memes and comment on news.",
        "Lurker turned commenter. Python dev.", "Hobbyist photographer. r/photography regular.",
        "Science enthusiast. PhD in Biology.", "Amateur chef. Love r/cooking.",
        "Finance nerd. Index fund investor.", "Book worm. r/books contributor.",
        "Fitness addict. Marathon runner.", "Board game collector. r/boardgames mod.",
        "Astronomy hobbyist. r/space regular.", "PC gamer. r/buildapc helper.",
    ]
    fake_abouts = [
        "", "", "Making money online. DM for info.",
        "Crypto trader. Free signals in bio.", "OnlyFans promo. 18+",
        "Forex expert. 99% win rate.", "Earn $500/day from home.",
    ]

    rows = []

    # Legit: variety of account ages and karma levels
    for _ in range(n_legit):
        age_days = int(RNG.integers(90, 5000))
        total_k  = int(RNG.integers(100, 500_000))
        ck = int(total_k * RNG.uniform(0.5, 0.95))
        pk = total_k - ck
        rows.append({
            "username":      rand_username(1, "legit")[0],
            "karma":         total_k,
            "comment_karma": ck,
            "post_karma":    pk,
            "created_at":    (base - timedelta(days=age_days)).strftime("%Y-%m-%d"),
            "about":         RNG.choice(legit_abouts),
            "label": 0,
        })

    # Fake: new accounts, minimal karma
    for _ in range(n_fake):
        age_days = int(RNG.integers(0, 90))
        rows.append({
            "username":      rand_username(1, "fake")[0],
            "karma":         int(RNG.integers(0, 80)),
            "comment_karma": int(RNG.integers(0, 50)),
            "post_karma":    int(RNG.integers(0, 30)),
            "created_at":    (base - timedelta(days=age_days)).strftime("%Y-%m-%d"),
            "about":         RNG.choice(fake_abouts),
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "reddit_train.csv", index=False)
    print(f"Reddit:    {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── 10. Snapchat ──────────────────────────────────────────────────────────────

def gen_snapchat(n_fake=1000, n_legit=1000):
    rows = []

    for _ in range(n_legit):
        score = int(sample(1, 500, 800_000, "log")[0])
        rows.append({
            "username":      rand_username(1, "legit")[0],
            "score":         score,
            "bio":           rand_bio(1, "legit")[0],
            "friends_count": int(RNG.integers(10, 1000)),
            "streaks":       int(RNG.integers(0, 2500)),
            "label": 0,
        })

    for _ in range(n_fake):
        rows.append({
            "username":      rand_username(1, "fake")[0],
            "score":         int(RNG.integers(0, 300)),
            "bio":           rand_bio(1, "fake")[0],
            "friends_count": int(RNG.integers(0, 15)),
            "streaks":       0,
            "label": 1,
        })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT_DIR / "snapchat_train.csv", index=False)
    print(f"Snapchat:  {len(df)} rows  Fake={df.label.sum()}  Legit={(df.label==0).sum()}")
    return df


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating enhanced synthetic training datasets...\n")
    gen_instagram(1200, 1200)
    gen_facebook(1000, 1000)
    gen_x(1400, 1400)
    gen_linkedin(1000, 1000)
    gen_github(1000, 1200)
    gen_discord(1000, 1000)
    gen_youtube(1000, 1000)
    gen_tiktok(1200, 1200)
    gen_reddit(1200, 1200)
    gen_snapchat(1000, 1000)
    print(f"\nAll datasets saved to {OUT_DIR}/")
