"""
features/username_forensics.py — Username pattern analysis for fake profile detection.

Provides analyze_username() which extracts linguistic, structural, and entropy-based
signals from a username string. All features degrade gracefully when optional
libraries are unavailable.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: rapidfuzz for Levenshtein / brand similarity
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import distance as _rfdist
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.debug("rapidfuzz not available — brand similarity scoring disabled.")

# ---------------------------------------------------------------------------
# Optional dependency: NLTK English word corpus
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import words as _nltk_words
    _ENGLISH_WORDS: Optional[set] = None  # lazy-loaded

    def _get_english_words() -> set:
        global _ENGLISH_WORDS
        if _ENGLISH_WORDS is None:
            try:
                _ENGLISH_WORDS = set(w.lower() for w in _nltk_words.words())
            except Exception:
                import nltk
                nltk.download("words", quiet=True)
                _ENGLISH_WORDS = set(w.lower() for w in _nltk_words.words())
        return _ENGLISH_WORDS

    NLTK_WORDS_AVAILABLE = True
except ImportError:
    NLTK_WORDS_AVAILABLE = False
    logger.debug("nltk.corpus.words not available — dictionary coverage disabled.")

# ---------------------------------------------------------------------------
# Bot-pattern regexes (compiled once at import time)
# ---------------------------------------------------------------------------
_BOT_PATTERNS = [
    re.compile(r"^[A-Z][a-z]+[A-Z][a-z]+\d{3,}$"),       # CamelCaseWord + 3+ digits
    re.compile(r"^[a-z]+_[a-z]+_\d+$"),                    # word_word_digits
    re.compile(r"^[a-zA-Z]{2,4}\d{6,}$"),                  # short prefix + 6+ digits
    re.compile(r"^\d{8,}$"),                                # purely numeric (8+ digits)
    re.compile(r"^[a-z]{1,3}\d{4,}[a-z]{0,3}$"),           # very short prefix + 4+ digits
    re.compile(r"^(user|acc|account|bot|fake|spam)\d+", re.IGNORECASE),  # generic prefixes
]

# Homoglyph map: Unicode look-alikes → ASCII equivalent
_HOMOGLYPHS: Dict[str, str] = {
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "p",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0445": "x",  # Cyrillic х
    "\u0456": "i",  # Cyrillic і (Ukrainian)
    "\u0391": "A",  # Greek Α
    "\u0395": "E",  # Greek Ε
    "\u039f": "O",  # Greek Ο
    "\u03a1": "P",  # Greek Ρ
    "\u0392": "B",  # Greek Β
    "\u0422": "T",  # Cyrillic Т
    "\u041c": "M",  # Cyrillic М
    "\u0417": "3",  # Cyrillic З
    "\u0406": "I",  # Ukrainian І
}

# ---------------------------------------------------------------------------
# Brand/celebrity names (loaded from data file if present, else inline fallback)
# ---------------------------------------------------------------------------
_BRAND_NAMES: Optional[list] = None


def _get_brand_names() -> list:
    """Load brand names from data/brand_names.txt or return a minimal fallback."""
    global _BRAND_NAMES
    if _BRAND_NAMES is not None:
        return _BRAND_NAMES

    from pathlib import Path
    brand_file = Path(__file__).parent.parent / "data" / "brand_names.txt"
    if brand_file.exists():
        _BRAND_NAMES = [
            line.strip().lower() for line in brand_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        _BRAND_NAMES = [
            "google", "apple", "microsoft", "amazon", "facebook", "instagram",
            "twitter", "youtube", "netflix", "tesla", "nike", "adidas", "amazon",
            "paypal", "binance", "coinbase", "opensea", "discord", "reddit",
            "elon", "musk", "trump", "biden", "obama", "oprah", "taylor", "swift",
            "beyonce", "rihanna", "cristiano", "ronaldo", "messi",
        ]
    return _BRAND_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of character distribution in string s."""
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values())


def _is_pronounceable(username: str) -> float:
    """
    Heuristic pronounceability score (0-1).
    Checks for alternating vowel/consonant patterns typical of real words.
    """
    s = re.sub(r"[^a-zA-Z]", "", username.lower())
    if len(s) < 2:
        return 0.5
    VOWELS = set("aeiou")
    transitions = 0
    total = max(len(s) - 1, 1)
    for i in range(len(s) - 1):
        if (s[i] in VOWELS) != (s[i + 1] in VOWELS):
            transitions += 1
    return transitions / total


def _longest_consonant_cluster(s: str) -> int:
    """Return the length of the longest consecutive consonant sequence."""
    VOWELS = set("aeiou")
    alpha = re.sub(r"[^a-z]", "", s.lower())
    if not alpha:
        return 0
    max_run = cur_run = 0
    for c in alpha:
        if c not in VOWELS:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def _detect_homoglyphs(username: str) -> int:
    """Count non-ASCII / homoglyph characters in username."""
    count = 0
    for ch in username:
        if ord(ch) > 127 or ch in _HOMOGLYPHS:
            count += 1
    # Also count zero-width / invisible characters
    for ch in username:
        cat = unicodedata.category(ch)
        if cat in ("Cf", "Cs", "Co"):  # Format, Surrogate, Private use
            count += 1
    return count


def _min_brand_distance(username: str) -> tuple[float, str]:
    """
    Return (min_levenshtein_distance, closest_brand_name).
    Uses rapidfuzz if available, else a simple character comparison.
    """
    brands = _get_brand_names()
    uname_lower = username.lower()

    if RAPIDFUZZ_AVAILABLE:
        best_dist = float("inf")
        best_name = ""
        for brand in brands:
            d = _rfdist.Levenshtein.distance(uname_lower, brand)
            if d < best_dist:
                best_dist = d
                best_name = brand
        return float(best_dist), best_name
    else:
        # Fallback: substring containment check
        for brand in brands:
            if brand in uname_lower or uname_lower in brand:
                return 0.0, brand
        return 99.0, ""


def _dictionary_coverage(username: str) -> float:
    """
    Fraction of word-like segments (split by digits/underscores/dots/dashes)
    that appear in the English dictionary.
    """
    if not NLTK_WORDS_AVAILABLE:
        return 0.5  # neutral default

    words_dict = _get_english_words()
    segments = re.split(r"[\d_.\-]+", username.lower())
    segments = [s for s in segments if len(s) >= 2]
    if not segments:
        return 0.0
    matches = sum(1 for s in segments if s in words_dict)
    return matches / len(segments)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_RESULT: Dict[str, object] = {
    "uname_length": None,
    "uname_digit_ratio": None,
    "uname_digit_suffix_length": None,
    "uname_consonant_cluster_ratio": None,
    "uname_vowel_ratio": None,
    "uname_is_pronounceable": None,
    "uname_bot_pattern_count": None,
    "uname_special_char_count": None,
    "uname_entropy": None,
    "uname_dict_coverage": None,
    "uname_homoglyph_count": None,
    "uname_brand_distance": None,
    "uname_brand_closest": None,
}


def analyze_username(username: Optional[str]) -> Dict[str, object]:
    """
    Analyze a username string and return a dict of feature_name -> value.

    All features are set to None on empty/None input.
    Features:
        uname_length                — character count
        uname_digit_ratio           — digits / total chars
        uname_digit_suffix_length   — count of trailing digit characters
        uname_consonant_cluster_ratio — longest consonant run / alpha chars
        uname_vowel_ratio           — vowels / alphabetic chars
        uname_is_pronounceable      — V/C alternation score (0-1)
        uname_bot_pattern_count     — number of bot-pattern regexes matched
        uname_special_char_count    — underscores + dots + dashes
        uname_entropy               — Shannon entropy of character distribution
        uname_dict_coverage         — fraction of segments in English dictionary
        uname_homoglyph_count       — suspicious Unicode / homoglyph chars
        uname_brand_distance        — min Levenshtein distance to known brands
        uname_brand_closest         — closest brand name matched
    """
    if not username or not isinstance(username, str) or not username.strip():
        return dict(_DEFAULT_RESULT)

    u = username.strip()

    try:
        length = len(u)
        digits = sum(c.isdigit() for c in u)
        alpha = sum(c.isalpha() for c in u)
        vowels = sum(c.lower() in "aeiou" for c in u if c.isalpha())
        specials = u.count("_") + u.count(".") + u.count("-")

        # Trailing digit length
        suffix_match = re.search(r"\d+$", u)
        digit_suffix_len = len(suffix_match.group()) if suffix_match else 0

        # Consonant cluster
        cluster = _longest_consonant_cluster(u)
        cluster_ratio = cluster / max(alpha, 1)

        # Bot patterns
        bot_hits = sum(1 for p in _BOT_PATTERNS if p.search(u))

        # Brand distance
        brand_dist, brand_closest = _min_brand_distance(u)

        return {
            "uname_length": length,
            "uname_digit_ratio": digits / max(length, 1),
            "uname_digit_suffix_length": digit_suffix_len,
            "uname_consonant_cluster_ratio": cluster_ratio,
            "uname_vowel_ratio": vowels / max(alpha, 1),
            "uname_is_pronounceable": _is_pronounceable(u),
            "uname_bot_pattern_count": bot_hits,
            "uname_special_char_count": specials,
            "uname_entropy": _shannon_entropy(u),
            "uname_dict_coverage": _dictionary_coverage(u),
            "uname_homoglyph_count": _detect_homoglyphs(u),
            "uname_brand_distance": brand_dist,
            "uname_brand_closest": brand_closest,
        }

    except Exception as exc:
        logger.warning("analyze_username failed for '%s': %s", username, exc)
        return dict(_DEFAULT_RESULT)
