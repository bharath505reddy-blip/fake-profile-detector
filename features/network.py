"""
features/network.py — Network and graph-based feature extraction.

Computes follow graph topology features including clustering, community detection,
mutual connections, and fake-neighbor ratios.

All third-party imports (networkx, community) are optional and fail gracefully.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx unavailable — graph features will be 0.0")

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    try:
        from community import community_louvain
        LOUVAIN_AVAILABLE = True
    except ImportError:
        LOUVAIN_AVAILABLE = False


def _mutual_follower_ratio(followers: List, following: List) -> float:
    """
    Ratio of mutual (bidirectional) connections to total unique connections.
    Formula: |followers ∩ following| / |followers ∪ following|
    """
    if not followers and not following:
        return 0.0
    f_set = set(str(x) for x in followers)
    g_set = set(str(x) for x in following)
    intersection = len(f_set & g_set)
    union = len(f_set | g_set)
    return round(intersection / max(union, 1), 4)


def _follower_quality_score(follower_metadata: List[dict]) -> float:
    """
    Average quality of followers based on lightweight heuristics:
    quality = (account_age_days / 365) * has_avatar * has_bio, clipped to 0-1.
    Higher score = higher-quality followers = more legit.
    """
    if not follower_metadata:
        return 0.0
    scores = []
    for meta in follower_metadata:
        age_years = min(float(meta.get("account_age_days") or 0) / 365.0, 1.0)
        has_avatar = int(bool(meta.get("has_avatar", 0)))
        has_bio = int(bool(meta.get("has_bio", 0)))
        score = age_years * (0.5 + 0.25 * has_avatar + 0.25 * has_bio)
        scores.append(min(score, 1.0))
    return round(float(sum(scores) / max(len(scores), 1)), 4)


def _reciprocity_rate(followers: List, following: List) -> float:
    """
    Fraction of accounts this profile follows that also follow back.
    High reciprocity is typical of organic networks.
    """
    if not following:
        return 0.0
    f_set = set(str(x) for x in followers)
    g_set = set(str(x) for x in following)
    reciprocal = len(f_set & g_set)
    return round(reciprocal / max(len(g_set), 1), 4)


def _celebrity_follow_ratio(verified_following_count: int, following_list: List) -> float:
    """
    Ratio of followed accounts that are verified/high-follower.
    Bots often follow many celebrities/verified accounts.
    """
    total_following = len(following_list) if following_list else 0
    return round(verified_following_count / max(total_following, 1), 4)


def _build_graph(followers: List, following: List, self_node: str = "self") -> Optional[object]:
    """
    Build a directed networkx graph from followers and following lists.
    Edges: follower -> self, self -> following.
    Returns None if networkx unavailable.
    """
    if not NETWORKX_AVAILABLE:
        return None
    G = nx.DiGraph()
    G.add_node(self_node)
    for f in followers:
        G.add_edge(str(f), self_node)
    for f in following:
        G.add_edge(self_node, str(f))
    return G


def _cluster_coefficient(followers: List, following: List) -> float:
    """
    Local clustering coefficient of the profile node in its follow network.
    Uses networkx. Returns 0 if unavailable.
    """
    if not NETWORKX_AVAILABLE:
        return 0.0
    try:
        G = _build_graph(followers, following)
        if G is None or "self" not in G:
            return 0.0
        undirected = G.to_undirected()
        return round(nx.clustering(undirected, "self"), 4)
    except Exception as exc:
        logger.debug("cluster_coefficient error: %s", exc)
        return 0.0


def _same_community_ratio(followers: List, following: List) -> float:
    """
    Fraction of connections in the same Louvain community as the profile.
    Very high ratio = suspicious coordination.
    Requires networkx + python-louvain.
    """
    if not NETWORKX_AVAILABLE or not LOUVAIN_AVAILABLE:
        return 0.0
    try:
        G = _build_graph(followers, following)
        if G is None:
            return 0.0
        undirected = G.to_undirected()
        if len(undirected.nodes) < 3:
            return 0.0
        partition = community_louvain.best_partition(undirected)
        self_community = partition.get("self", -1)
        neighbors = set(str(f) for f in followers) | set(str(f) for f in following)
        if not neighbors:
            return 0.0
        same_count = sum(
            1 for n in neighbors if partition.get(n, -2) == self_community
        )
        return round(same_count / len(neighbors), 4)
    except Exception as exc:
        logger.debug("same_community_ratio error: %s", exc)
        return 0.0


def _fake_neighbor_ratio(
    followers: List, following: List,
    existing_predictions: Optional[Dict[str, str]]
) -> float:
    """
    Fraction of direct connections already labeled as "Fake" by the model.
    This is a powerful second-pass enrichment feature.
    Returns 0 if no existing_predictions provided.
    """
    if not existing_predictions:
        return 0.0
    neighbors = set(str(f) for f in followers) | set(str(f) for f in following)
    if not neighbors:
        return 0.0
    fake_count = sum(
        1 for n in neighbors
        if existing_predictions.get(n, "Legit") == "Fake"
    )
    return round(fake_count / len(neighbors), 4)


def extract_network_features(
    profile_data: dict,
    existing_predictions: Optional[Dict[str, str]] = None
) -> dict:
    """
    Extract network and graph-based features from a profile data dict.

    Expected profile_data keys (all optional):
      followers_list          — list of follower usernames/ids
      following_list          — list of following usernames/ids
      follower_metadata       — list of dicts: {account_age_days, has_avatar, has_bio}
      verified_following_count — int (how many followed accounts are verified)

    existing_predictions: dict mapping username -> "Fake"/"Legit" for second-pass enrichment.

    Returns dict with keys:
      mutual_follower_ratio, follower_quality_score, reciprocity_rate,
      celebrity_follow_ratio, cluster_coefficient, same_community_ratio,
      fake_neighbor_ratio
    """
    features: Dict[str, float] = {}

    try:
        followers = profile_data.get("followers_list") or []
        following = profile_data.get("following_list") or []
        follower_metadata = profile_data.get("follower_metadata") or []
        verified_following_count = int(profile_data.get("verified_following_count") or 0)

        features["mutual_follower_ratio"] = _mutual_follower_ratio(followers, following)
        features["follower_quality_score"] = _follower_quality_score(follower_metadata)
        features["reciprocity_rate"] = _reciprocity_rate(followers, following)
        features["celebrity_follow_ratio"] = _celebrity_follow_ratio(verified_following_count, following)
        features["cluster_coefficient"] = _cluster_coefficient(followers, following)
        features["same_community_ratio"] = _same_community_ratio(followers, following)
        features["fake_neighbor_ratio"] = _fake_neighbor_ratio(followers, following, existing_predictions)

    except Exception as exc:
        logger.warning("extract_network_features error: %s", exc)

    # Replace NaN/inf with 0.0
    for k, v in features.items():
        try:
            if not math.isfinite(float(v)):
                features[k] = 0.0
        except (TypeError, ValueError):
            features[k] = 0.0

    return features
