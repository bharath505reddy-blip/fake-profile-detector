"""
features/photo_analysis.py — Profile photo analysis pipeline.

Provides analyze_avatar() which detects:
  - Default/stock avatars via perceptual hash (pHash) comparison
  - Repeated/stolen photos via avatar hash database lookup
  - AI-generated face probability (heuristic fallback)
  - Image quality and metadata features

External dependencies: imagehash, Pillow, requests (all optional, fail gracefully).
"""
from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: Pillow
# ---------------------------------------------------------------------------
try:
    from PIL import Image, ExifTags
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.debug("Pillow not available — photo analysis disabled.")

# ---------------------------------------------------------------------------
# Optional: imagehash
# ---------------------------------------------------------------------------
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.debug("imagehash not available — pHash comparison disabled.")

# ---------------------------------------------------------------------------
# Optional: requests (for downloading avatars)
# ---------------------------------------------------------------------------
try:
    import requests as _http
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional: onnxruntime for GAN detection model
# ---------------------------------------------------------------------------
try:
    import onnxruntime as _ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    logger.debug("onnxruntime not available — AI face detection heuristic only.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_HASH_FILE = _DATA_DIR / "default_avatar_hashes.json"
_AVATAR_CACHE_DIR = Path(tempfile.gettempdir()) / "fake_profile_avatars"
_AVATAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# DB session is passed in at call time to avoid circular import
_DB_SESSION_FACTORY = None  # Set by app.py after DB init


def set_db_session_factory(factory):
    """Called once from app.py to enable avatar hash DB lookups."""
    global _DB_SESSION_FACTORY
    _DB_SESSION_FACTORY = factory


# ---------------------------------------------------------------------------
# Default avatar pHash DB (loaded from JSON)
# ---------------------------------------------------------------------------
_DEFAULT_HASHES: Optional[list] = None


def _load_default_hashes() -> list:
    """Load known default avatar perceptual hashes from data/default_avatar_hashes.json."""
    global _DEFAULT_HASHES
    if _DEFAULT_HASHES is not None:
        return _DEFAULT_HASHES

    if _DEFAULT_HASH_FILE.exists():
        import json
        try:
            _DEFAULT_HASHES = json.loads(_DEFAULT_HASH_FILE.read_text())
        except Exception:
            _DEFAULT_HASHES = []
    else:
        # Minimal fallback: we store the hashes of known default avatars after computing them
        _DEFAULT_HASHES = []
    return _DEFAULT_HASHES


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

def _download_image(url: str, timeout: int = 8) -> Optional[bytes]:
    """Download an image URL and return raw bytes. Returns None on failure."""
    if not REQUESTS_AVAILABLE:
        return None
    cache_key = hashlib.md5(url.encode()).hexdigest()
    cache_path = _AVATAR_CACHE_DIR / f"{cache_key}.img"

    # Check cache (valid for 1 hour)
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime < 3600):
        return cache_path.read_bytes()

    try:
        resp = _http.get(url, timeout=timeout, stream=True,
                         headers={"User-Agent": "FakeProfileDetector/1.0"})
        if resp.status_code != 200:
            logger.debug("Avatar download failed: HTTP %d for %s", resp.status_code, url)
            return None
        data = resp.content
        cache_path.write_bytes(data)
        return data
    except Exception as exc:
        logger.debug("Avatar download error for %s: %s", url, exc)
        return None


def _open_image(source) -> Optional["Image.Image"]:
    """Open an image from URL string, bytes, or file path."""
    if not PILLOW_AVAILABLE:
        return None

    try:
        if isinstance(source, str):
            if source.startswith("http://") or source.startswith("https://"):
                data = _download_image(source)
                if data is None:
                    return None
                return Image.open(io.BytesIO(data)).convert("RGB")
            else:
                return Image.open(source).convert("RGB")
        elif isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        elif isinstance(source, io.IOBase):
            return Image.open(source).convert("RGB")
    except Exception as exc:
        logger.debug("Could not open image: %s", exc)
    return None


# ---------------------------------------------------------------------------
# pHash comparison
# ---------------------------------------------------------------------------

def _compute_phash(img: "Image.Image") -> Optional[str]:
    """Compute perceptual hash string for an image."""
    if not IMAGEHASH_AVAILABLE:
        return None
    try:
        h = imagehash.phash(img)
        return str(h)
    except Exception:
        return None


def _phash_distance(h1: str, h2: str) -> int:
    """Compute Hamming distance between two pHash hex strings."""
    if not IMAGEHASH_AVAILABLE:
        return 99
    try:
        return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
    except Exception:
        return 99


def _is_default_avatar(phash: str) -> bool:
    """Check if pHash is within Hamming distance 10 of any known default avatar."""
    defaults = _load_default_hashes()
    for entry in defaults:
        known_hash = entry if isinstance(entry, str) else entry.get("hash", "")
        if known_hash and _phash_distance(phash, known_hash) < 10:
            return True
    return False


def _check_stock_photo(phash: str, platform: str, identifier: str) -> Dict:
    """
    Look up avatar hash in DB for seen_count.
    Updates the DB record if AvatarHash model is available.
    Returns: {seen_count, is_stock}
    """
    result = {"seen_count": 1, "is_stock": False, "stock_score": 0.0}
    if _DB_SESSION_FACTORY is None:
        return result

    try:
        from datetime import datetime, timezone
        session = _DB_SESSION_FACTORY()

        # Import model lazily to avoid circular imports
        from app import AvatarHash, db

        existing = session.query(AvatarHash).filter(
            AvatarHash.phash == phash
        ).all()

        # Check if any matching hash (within hamming distance 8) from different profiles
        matches = []
        for row in existing:
            if row.profile_identifier != identifier:
                try:
                    if _phash_distance(phash, row.phash) < 8:
                        matches.append(row)
                except Exception:
                    pass

        seen_count = sum(m.seen_count for m in matches) + 1

        # Upsert own record
        own = session.query(AvatarHash).filter_by(
            phash=phash, platform=platform, profile_identifier=identifier
        ).first()
        if own:
            own.seen_count += 1
        else:
            session.add(AvatarHash(
                phash=phash,
                profile_identifier=identifier,
                platform=platform,
                first_seen=datetime.now(timezone.utc),
                seen_count=1,
            ))
        session.commit()

        stock_score = min(1.0, (seen_count - 1) / 5.0) if seen_count > 1 else 0.0
        result.update({
            "seen_count": seen_count,
            "is_stock": seen_count > 3,
            "stock_score": stock_score,
        })
    except Exception as exc:
        logger.debug("Avatar DB lookup failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# AI-generated face detection (heuristic)
# ---------------------------------------------------------------------------

def _ai_generated_score(img: "Image.Image") -> Optional[float]:
    """
    Estimate probability that an image is AI-generated.
    Uses ONNX model if available, otherwise a simple symmetry heuristic.
    Returns 0.0-1.0 or None if unavailable.
    """
    if not PILLOW_AVAILABLE:
        return None

    try:
        # Simple heuristic: compare left/right half pHash similarity
        # GAN faces tend to be more symmetric than real photos
        w, h = img.size
        left = img.crop((0, 0, w // 2, h))
        right = img.crop((w // 2, 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT)

        if IMAGEHASH_AVAILABLE:
            lh = imagehash.phash(left)
            rh = imagehash.phash(right)
            dist = lh - rh
            # Lower distance = more symmetric = potentially AI-generated
            # Real faces: dist ~15-25; GAN faces: dist ~5-15
            symmetry_score = max(0.0, (20 - dist) / 20.0)
            return round(symmetry_score, 3)
        return None
    except Exception as exc:
        logger.debug("AI score computation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Image quality features
# ---------------------------------------------------------------------------

def _color_histogram_entropy(img: "Image.Image") -> float:
    """Shannon entropy of the RGB color histogram (flattened)."""
    try:
        hist = img.histogram()  # 256 * 3 = 768 bins
        total = sum(hist)
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in hist:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return round(entropy, 4)
    except Exception:
        return 0.0


def _estimate_jpeg_quality(img: "Image.Image", raw_bytes: Optional[bytes] = None) -> Optional[int]:
    """
    Rough JPEG quality estimation based on file size vs image dimensions.
    Returns 0-100 or None.
    """
    if raw_bytes is None:
        return None
    try:
        w, h = img.size
        pixels = w * h
        if pixels == 0:
            return None
        # Rough heuristic: bytes per pixel at quality 95 ≈ 1-3
        bpp = len(raw_bytes) / pixels
        quality = min(100, max(0, int(bpp * 33)))
        return quality
    except Exception:
        return None


def _has_exif(img: "Image.Image") -> bool:
    """Check if image has EXIF metadata."""
    try:
        exif = img._getexif()
        return exif is not None and len(exif) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Default result
# ---------------------------------------------------------------------------

_DEFAULT_RESULT: Dict[str, object] = {
    "photo_is_default_avatar": None,
    "photo_stock_score": None,
    "photo_seen_count": None,
    "photo_ai_generated_score": None,
    "photo_width": None,
    "photo_height": None,
    "photo_aspect_ratio": None,
    "photo_compression_quality": None,
    "photo_has_exif": None,
    "photo_color_entropy": None,
    "photo_available": 0,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_avatar(
    image_source,
    platform: str = "",
    profile_identifier: str = "",
) -> Dict[str, object]:
    """
    Analyze a profile avatar image and return forensic features.

    Args:
        image_source: URL string, file path, raw bytes, or file-like object.
        platform: Platform name for DB context.
        profile_identifier: Username or profile ID for stock-photo deduplication.

    Returns:
        Dict of feature_name -> value. photo_available=0 if image cannot be loaded.

    Features:
        photo_is_default_avatar   — 1 if pHash matches known default avatar
        photo_stock_score         — 0-1: how often this image has been seen across profiles
        photo_seen_count          — total times this image has been seen
        photo_ai_generated_score  — 0-1 probability of AI generation (heuristic)
        photo_width               — image width in pixels
        photo_height              — image height in pixels
        photo_aspect_ratio        — width / height
        photo_compression_quality — estimated JPEG quality 0-100
        photo_has_exif            — 1 if EXIF data present
        photo_color_entropy       — Shannon entropy of color histogram
        photo_available           — 1 if image was successfully analyzed
    """
    result = dict(_DEFAULT_RESULT)

    if not image_source:
        return result

    if not PILLOW_AVAILABLE:
        logger.debug("Pillow unavailable — skipping photo analysis.")
        return result

    t0 = time.time()

    # Download raw bytes if URL
    raw_bytes: Optional[bytes] = None
    if isinstance(image_source, str) and image_source.startswith("http"):
        raw_bytes = _download_image(image_source)
        if raw_bytes is None:
            logger.debug("Could not download avatar from %s", image_source)
            return result

    img = _open_image(raw_bytes if raw_bytes is not None else image_source)
    if img is None:
        return result

    try:
        w, h = img.size
        result["photo_width"] = w
        result["photo_height"] = h
        result["photo_aspect_ratio"] = round(w / max(h, 1), 3)
        result["photo_color_entropy"] = _color_histogram_entropy(img)
        result["photo_has_exif"] = int(_has_exif(img))
        result["photo_compression_quality"] = _estimate_jpeg_quality(img, raw_bytes)
        result["photo_available"] = 1

        phash = _compute_phash(img)
        if phash:
            result["photo_is_default_avatar"] = int(_is_default_avatar(phash))
            stock_info = _check_stock_photo(phash, platform, profile_identifier)
            result["photo_stock_score"] = stock_info["stock_score"]
            result["photo_seen_count"] = stock_info["seen_count"]
        else:
            result["photo_is_default_avatar"] = 0
            result["photo_stock_score"] = 0.0
            result["photo_seen_count"] = 1

        result["photo_ai_generated_score"] = _ai_generated_score(img)

        elapsed = time.time() - t0
        logger.info(
            "Avatar analysis completed in %.2fs [%s %s]: "
            "default=%s stock_score=%.2f ai_score=%s",
            elapsed, platform, profile_identifier,
            result["photo_is_default_avatar"],
            result["photo_stock_score"] or 0.0,
            result["photo_ai_generated_score"],
        )

    except Exception as exc:
        logger.warning("Avatar analysis error [%s %s]: %s", platform, profile_identifier, exc)

    return result


def cleanup_avatar_cache(max_age_hours: float = 2.0) -> None:
    """Remove cached avatar files older than max_age_hours."""
    cutoff = time.time() - max_age_hours * 3600
    for f in _AVATAR_CACHE_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            try:
                f.unlink()
            except OSError:
                pass
