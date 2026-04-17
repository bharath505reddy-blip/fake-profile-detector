"""
features/bio_forensics.py — Bio/description text forensic analysis.

Provides analyze_bio() which computes spam, template similarity, linguistic,
and structural signals from a profile bio or about text.

All external dependencies (langdetect, nltk, emoji) are optional and fail gracefully.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: langdetect
# ---------------------------------------------------------------------------
try:
    from langdetect import detect as _langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.debug("langdetect not available — language detection disabled.")

# ---------------------------------------------------------------------------
# Spam keyword lexicon
# ---------------------------------------------------------------------------
_SPAM_KEYWORDS = {
    "high": [
        "airdrop", "whitelist", "giveaway", "dm for", "send dm",
        "forex", "binary options", "passive income", "financial freedom",
        "click link", "limited spots", "act now", "earn daily",
        "free bitcoin", "free btc", "free eth", "free crypto",
        "get rich", "make money fast", "easy money", "100% profit",
        "99% win rate", "investment advisor", "earn while you sleep",
        "make $", "earn $", "official giveaway",
    ],
    "medium": [
        "crypto", "nft", "bitcoin", "btc", "ethereum", "eth", "defi", "web3",
        "trading", "invest", "profit", "roi", "hodl", "pump", "moon",
        "tokens", "altcoin", "binance", "coinbase",
    ],
    "low": [
        "follow back", "f4f", "l4l", "collab", "promo",
        "link in bio", "dm me", "inbox me",
    ],
}

# Invisible / zero-width Unicode characters
_INVISIBLE_CHARS = {
    "\u200b",  # Zero width space
    "\u200c",  # Zero width non-joiner
    "\u200d",  # Zero width joiner
    "\u200e",  # Left-to-right mark
    "\u200f",  # Right-to-left mark
    "\u202a",  # Left-to-right embedding
    "\u202b",  # Right-to-left embedding
    "\u202c",  # Pop directional formatting
    "\u202d",  # Left-to-right override
    "\u202e",  # Right-to-left override
    "\ufeff",  # BOM / zero-width no-break space
    "\u2060",  # Word joiner
    "\u2061",  # Function application
    "\u2062",  # Invisible times
    "\u2063",  # Invisible separator
}

# Phone number pattern
_PHONE_RE = re.compile(
    r"(\+?\d[\d\s\-\(\)]{7,}\d)"
)

# Email pattern
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# URL pattern
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Shortened URL domains
_SHORT_URL_DOMAINS = {
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "buff.ly",
    "dlvr.it", "ift.tt", "fb.me", "youtu.be", "amzn.to", "is.gd",
    "short.io", "shorturl.at", "rb.gy", "cutt.ly",
}

# Emoji Unicode ranges
def _count_emojis(text: str) -> int:
    count = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x1F300 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or 0xFE00 <= cp <= 0xFE0F
            or 0x1F000 <= cp <= 0x1F02F
        ):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Bio template DB (lazy loaded)
# ---------------------------------------------------------------------------
_BIO_TEMPLATES: Optional[List[str]] = None


def _get_bio_templates() -> List[str]:
    """Load known fake bio templates from data/bio_templates.json."""
    global _BIO_TEMPLATES
    if _BIO_TEMPLATES is not None:
        return _BIO_TEMPLATES

    tmpl_file = Path(__file__).parent.parent / "data" / "bio_templates.json"
    if tmpl_file.exists():
        try:
            _BIO_TEMPLATES = json.loads(tmpl_file.read_text())
        except Exception:
            _BIO_TEMPLATES = []
    else:
        _BIO_TEMPLATES = [
            "click link below earn money",
            "free bitcoin airdrop link in bio",
            "earn passive income daily dm me",
            "investment advisor forex signals dm",
            "follow for follow back f4f",
            "crypto signals 99 win rate",
            "make money online link below",
            "official giveaway account repost to win",
        ]
    return _BIO_TEMPLATES


def _trigram_overlap(a: str, b: str) -> float:
    """Character-level trigram Jaccard overlap between two strings."""
    def trigrams(s: str) -> set:
        s = s.lower()
        return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else set()

    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _template_similarity(bio: str) -> float:
    """Max trigram similarity between bio and known fake bio templates."""
    templates = _get_bio_templates()
    if not templates or not bio:
        return 0.0
    return max(_trigram_overlap(bio, t) for t in templates)


def _count_repeated_sequences(text: str) -> float:
    """
    Ratio of characters in runs of 3+ identical consecutive characters.
    e.g. "🔥🔥🔥🔥" or "!!!!!!!" contributes heavily.
    """
    if not text:
        return 0.0
    run_chars = 0
    i = 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == text[i]:
            j += 1
        run_len = j - i
        if run_len >= 3:
            run_chars += run_len
        i = j
    return run_chars / max(len(text), 1)


def _spam_score(text: str, word_count: int) -> float:
    """
    Weighted spam keyword score normalised by word count.
    """
    t = text.lower()
    high = sum(1 for kw in _SPAM_KEYWORDS["high"] if kw in t)
    med = sum(1 for kw in _SPAM_KEYWORDS["medium"] if kw in t)
    low = sum(1 for kw in _SPAM_KEYWORDS["low"] if kw in t)
    raw = 3 * high + 2 * med + 1 * low
    return raw / max(word_count, 1)


def _detect_invisible(text: str) -> int:
    """Count invisible/zero-width Unicode characters."""
    return sum(1 for ch in text if ch in _INVISIBLE_CHARS)


def _count_shortened_urls(text: str) -> int:
    """Count URLs whose domain matches known URL shorteners."""
    urls = _URL_RE.findall(text)
    count = 0
    for url in urls:
        for domain in _SHORT_URL_DOMAINS:
            if domain in url.lower():
                count += 1
                break
    return count


def _excessive_caps_ratio(text: str) -> float:
    """Uppercase / total alphabetic characters."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _detect_language(text: str) -> Optional[str]:
    """Detect ISO language code using langdetect; returns None on failure."""
    if not LANGDETECT_AVAILABLE or not text or len(text.strip()) < 10:
        return None
    try:
        return _langdetect(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Default result structure
# ---------------------------------------------------------------------------
_DEFAULT_RESULT: Dict[str, object] = {
    "bio_length": None,
    "bio_word_count": None,
    "bio_emoji_count": None,
    "bio_emoji_ratio": None,
    "bio_url_count": None,
    "bio_shortened_url_count": None,
    "bio_mention_count": None,
    "bio_hashtag_count": None,
    "bio_spam_score": None,
    "bio_template_similarity": None,
    "bio_language": None,
    "bio_invisible_char_count": None,
    "bio_phone_present": None,
    "bio_email_present": None,
    "bio_excessive_caps_ratio": None,
    "bio_repeated_char_ratio": None,
    "bio_is_empty": None,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_bio(bio_text: Optional[str], platform: str = "") -> Dict[str, object]:
    """
    Analyze a profile bio/description and return forensic features.

    Args:
        bio_text: Raw bio string (may be empty or None).
        platform: Platform name for context logging.

    Returns:
        Dict of feature_name -> value. All values are None on empty input.

    Features:
        bio_length                — character count
        bio_word_count            — word count
        bio_emoji_count           — number of emoji characters
        bio_emoji_ratio           — emojis / total chars
        bio_url_count             — URLs detected
        bio_shortened_url_count   — known URL shortener links
        bio_mention_count         — @mentions
        bio_hashtag_count         — #hashtags
        bio_spam_score            — weighted crypto/spam keyword density
        bio_template_similarity   — max trigram overlap with known fake bio templates
        bio_language              — detected ISO language code
        bio_invisible_char_count  — zero-width / invisible Unicode chars
        bio_phone_present         — boolean: phone number detected
        bio_email_present         — boolean: email detected
        bio_excessive_caps_ratio  — uppercase / total alpha chars
        bio_repeated_char_ratio   — ratio of chars in runs of 3+ identical chars
        bio_is_empty              — boolean: bio is empty
    """
    if not bio_text or not isinstance(bio_text, str):
        result = dict(_DEFAULT_RESULT)
        result["bio_is_empty"] = 1
        result["bio_length"] = 0
        return result

    text = bio_text  # keep original for some checks
    stripped = text.strip()

    if not stripped:
        result = dict(_DEFAULT_RESULT)
        result["bio_is_empty"] = 1
        result["bio_length"] = 0
        return result

    try:
        words = stripped.split()
        word_count = len(words)
        length = len(stripped)
        emoji_count = _count_emojis(stripped)
        urls = _URL_RE.findall(stripped)
        url_count = len(urls)
        shortened_url_count = _count_shortened_urls(stripped)
        mention_count = len(re.findall(r"@\w+", stripped))
        hashtag_count = len(re.findall(r"#\w+", stripped))

        return {
            "bio_length": length,
            "bio_word_count": word_count,
            "bio_emoji_count": emoji_count,
            "bio_emoji_ratio": emoji_count / max(length, 1),
            "bio_url_count": url_count,
            "bio_shortened_url_count": shortened_url_count,
            "bio_mention_count": mention_count,
            "bio_hashtag_count": hashtag_count,
            "bio_spam_score": _spam_score(stripped, word_count),
            "bio_template_similarity": _template_similarity(stripped),
            "bio_language": _detect_language(stripped),
            "bio_invisible_char_count": _detect_invisible(stripped),
            "bio_phone_present": int(bool(_PHONE_RE.search(stripped))),
            "bio_email_present": int(bool(_EMAIL_RE.search(stripped))),
            "bio_excessive_caps_ratio": _excessive_caps_ratio(stripped),
            "bio_repeated_char_ratio": _count_repeated_sequences(stripped),
            "bio_is_empty": 0,
        }

    except Exception as exc:
        logger.warning("[%s] analyze_bio failed: %s", platform, exc)
        return dict(_DEFAULT_RESULT)
