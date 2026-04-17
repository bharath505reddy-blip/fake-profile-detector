"""
features/content.py — Text, linguistic, and content-based feature extraction.

All functions accept a profile_data dict and return a dict of feature_name -> float.
All third-party imports (sklearn, textblob, simhash) are optional and fail gracefully.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import simhash as _simhash_mod
    SIMHASH_AVAILABLE = True
except ImportError:
    SIMHASH_AVAILABLE = False


# Simple positive/negative wordlists for fallback sentiment
_POSITIVE_WORDS = {
    "good", "great", "love", "happy", "excellent", "amazing", "wonderful",
    "best", "awesome", "fantastic", "beautiful", "positive", "success",
    "win", "perfect", "nice", "brilliant", "enjoy", "thanks", "thank"
}
_NEGATIVE_WORDS = {
    "bad", "hate", "terrible", "awful", "worst", "horrible", "disgusting",
    "fail", "failure", "sad", "angry", "disappointed", "ugly", "poor",
    "wrong", "broken", "false", "spam", "scam", "fake", "fraud"
}


def _count_emojis(text: str) -> int:
    """Count emoji characters using unicode ranges."""
    count = 0
    for char in text:
        cp = ord(char)
        if (0x1F300 <= cp <= 0x1FFFF or
                0x2600 <= cp <= 0x27BF or
                0xFE00 <= cp <= 0xFE0F or
                0x1F1E0 <= cp <= 0x1F1FF):
            count += 1
    return count


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer returning lowercase non-empty tokens."""
    return [t.lower() for t in re.split(r"\s+", text.strip()) if t]


def _simple_sentiment(text: str) -> float:
    """
    Fallback sentiment: returns value in [-1, 1] based on positive/negative word counts.
    """
    tokens = set(_tokenize(text))
    pos = len(tokens & _POSITIVE_WORDS)
    neg = len(tokens & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _duplicate_content_ratio_sklearn(posts: List[str]) -> float:
    """
    Compute pairwise TF-IDF cosine similarity and return fraction of pairs with similarity > 0.85.
    """
    if len(posts) < 2:
        return 0.0
    try:
        vec = TfidfVectorizer(min_df=1, stop_words="english")
        mat = vec.fit_transform(posts)
        sims = cosine_similarity(mat)
        n = len(posts)
        high_sim_pairs = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                if sims[i, j] > 0.85:
                    high_sim_pairs += 1
        return round(high_sim_pairs / max(total_pairs, 1), 4)
    except Exception as exc:
        logger.debug("TF-IDF duplicate ratio error: %s", exc)
        return 0.0


def _duplicate_content_ratio_jaccard(posts: List[str]) -> float:
    """
    Fallback duplicate detection using Jaccard similarity on word sets.
    """
    if len(posts) < 2:
        return 0.0
    sets = [set(_tokenize(p)) for p in posts]
    n = len(sets)
    high_sim_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union > 0 and inter / union > 0.7:
                high_sim_pairs += 1
    return round(high_sim_pairs / max(total_pairs, 1), 4)


def _simhash_cluster_count(posts: List[str]) -> int:
    """
    Cluster posts by simhash fingerprint. Fewer clusters = more duplication.
    Falls back to Jaccard-based grouping if simhash unavailable.
    """
    if not posts:
        return 0
    if SIMHASH_AVAILABLE:
        try:
            from simhash import Simhash
            fingerprints = [Simhash(p).value for p in posts]
            clusters: List[List[int]] = []
            used = set()
            for i, fp_i in enumerate(fingerprints):
                if i in used:
                    continue
                cluster = [i]
                used.add(i)
                for j in range(i + 1, len(fingerprints)):
                    if j not in used:
                        dist = bin(fp_i ^ fingerprints[j]).count("1")
                        if dist <= 3:
                            cluster.append(j)
                            used.add(j)
                clusters.append(cluster)
            return len(clusters)
        except Exception as exc:
            logger.debug("simhash cluster error: %s", exc)

    # Jaccard fallback
    sets = [set(_tokenize(p)) for p in posts]
    clusters: List[int] = [-1] * len(sets)
    cluster_id = 0
    for i in range(len(sets)):
        if clusters[i] == -1:
            clusters[i] = cluster_id
            for j in range(i + 1, len(sets)):
                if clusters[j] == -1:
                    inter = len(sets[i] & sets[j])
                    union = len(sets[i] | sets[j])
                    if union > 0 and inter / union > 0.6:
                        clusters[j] = cluster_id
            cluster_id += 1
    return cluster_id


def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two strings using TF-IDF."""
    if not text_a or not text_b:
        return 0.0
    if not SKLEARN_AVAILABLE:
        # Jaccard fallback
        a_set = set(_tokenize(text_a))
        b_set = set(_tokenize(text_b))
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return round(inter / union, 4) if union > 0 else 0.0
    try:
        vec = TfidfVectorizer(min_df=1)
        mat = vec.fit_transform([text_a, text_b])
        sim = cosine_similarity(mat[0], mat[1])[0][0]
        return round(float(sim), 4)
    except Exception:
        return 0.0


def extract_content_features(profile_data: dict) -> dict:
    """
    Extract text and linguistic features from profile content.

    Expected profile_data keys:
      posts   — list of post text strings
      bio     — bio/about text string

    Returns dict with keys:
      duplicate_content_ratio, unique_content_ratio, simhash_cluster_count,
      vocabulary_richness, avg_sentence_length, sentiment_variance,
      sentiment_mean, subjectivity_mean, emoji_ratio, hashtag_ratio,
      caps_ratio, bio_post_topic_similarity, url_spam_ratio,
      mention_spam_ratio
    """
    features: Dict[str, float] = {}

    try:
        raw_posts = profile_data.get("posts") or []
        if isinstance(raw_posts, str):
            raw_posts = [raw_posts]
        posts: List[str] = [str(p) for p in raw_posts if p and str(p).strip()]
        bio: str = str(profile_data.get("bio") or "").strip()

        all_text = " ".join(posts)
        total_posts = len(posts)

        # --- Duplicate Content ---
        if SKLEARN_AVAILABLE:
            dup_ratio = _duplicate_content_ratio_sklearn(posts)
        else:
            dup_ratio = _duplicate_content_ratio_jaccard(posts)
        features["duplicate_content_ratio"] = dup_ratio
        features["unique_content_ratio"] = round(1.0 - dup_ratio, 4)
        features["simhash_cluster_count"] = float(_simhash_cluster_count(posts))

        # --- Vocabulary Richness ---
        words = _tokenize(all_text)
        total_words = len(words)
        unique_words = len(set(words))
        features["vocabulary_richness"] = round(unique_words / max(total_words, 1), 4)

        # --- Avg sentence length ---
        sentences = re.split(r"[.!?]+", all_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sent_len = np.mean([len(_tokenize(s)) for s in sentences])
            features["avg_sentence_length"] = round(float(avg_sent_len), 4)
        else:
            features["avg_sentence_length"] = 0.0

        # --- Sentiment ---
        if TEXTBLOB_AVAILABLE and posts:
            try:
                polarities = []
                subjectivities = []
                for post in posts:
                    blob = TextBlob(post)
                    polarities.append(blob.sentiment.polarity)
                    subjectivities.append(blob.sentiment.subjectivity)
                features["sentiment_variance"] = round(float(np.var(polarities)), 4)
                features["sentiment_mean"] = round(float(np.mean(polarities)), 4)
                features["subjectivity_mean"] = round(float(np.mean(subjectivities)), 4)
            except Exception as exc:
                logger.debug("TextBlob sentiment error: %s", exc)
                features["sentiment_variance"] = 0.0
                features["sentiment_mean"] = 0.0
                features["subjectivity_mean"] = 0.0
        else:
            # Fallback simple sentiment
            if posts:
                polarities = [_simple_sentiment(p) for p in posts]
                features["sentiment_variance"] = round(float(np.var(polarities)), 4)
                features["sentiment_mean"] = round(float(np.mean(polarities)), 4)
            else:
                features["sentiment_variance"] = 0.0
                features["sentiment_mean"] = 0.0
            features["subjectivity_mean"] = 0.0

        # --- Emoji ratio ---
        total_chars = len(all_text)
        emoji_count = _count_emojis(all_text)
        features["emoji_ratio"] = round(emoji_count / max(total_words, 1), 4)

        # --- Hashtag ratio ---
        hashtag_count = len(re.findall(r"#\w+", all_text))
        features["hashtag_ratio"] = round(hashtag_count / max(total_words, 1), 4)

        # --- Caps ratio ---
        upper_count = sum(1 for c in all_text if c.isupper())
        features["caps_ratio"] = round(upper_count / max(total_chars, 1), 4)

        # --- Bio-post topic similarity ---
        features["bio_post_topic_similarity"] = _tfidf_cosine_similarity(bio, all_text)

        # --- URL spam ratio ---
        url_pattern = re.compile(r"https?://|www\.|bit\.ly|tinyurl|t\.co", re.I)
        url_posts = sum(1 for p in posts if url_pattern.search(p))
        features["url_spam_ratio"] = round(url_posts / max(total_posts, 1), 4)

        # --- Mention spam ratio ---
        def _mention_heavy(post: str) -> bool:
            tokens = _tokenize(post)
            if not tokens:
                return False
            mention_count = sum(1 for t in tokens if t.startswith("@"))
            return mention_count / len(tokens) > 0.30

        mention_spam_posts = sum(1 for p in posts if _mention_heavy(p))
        features["mention_spam_ratio"] = round(mention_spam_posts / max(total_posts, 1), 4)

    except Exception as exc:
        logger.warning("extract_content_features error: %s", exc)

    # Replace NaN/inf with 0.0
    for k, v in features.items():
        try:
            if not math.isfinite(float(v)):
                features[k] = 0.0
        except (TypeError, ValueError):
            features[k] = 0.0

    return features
