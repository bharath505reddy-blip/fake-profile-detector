"""
feedback.py — User feedback system for fake profile predictions.

Stores user corrections to model predictions in SQLite.
Enables retraining with corrected labels via the /feedback/review admin page.

Usage in app.py:
    from feedback import init_feedback, save_feedback, get_pending_reviews, get_feedback_stats
    FeedbackEntry = init_feedback(db)
    with app.app_context():
        db.create_all()
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def init_feedback(db):
    """
    Initialize the FeedbackEntry SQLAlchemy model using the provided db instance.

    Call once from app.py after db is created:
        FeedbackEntry = init_feedback(db)
        with app.app_context():
            db.create_all()

    Returns:
        The FeedbackEntry SQLAlchemy model class.
    """

    class FeedbackEntry(db.Model):
        """Stores user corrections to model predictions for retraining."""

        __tablename__ = "feedback"

        id = db.Column(db.Integer, primary_key=True)
        platform = db.Column(db.String(50), nullable=False, index=True)
        profile_data_json = db.Column(db.Text, nullable=False)
        predicted_label = db.Column(db.String(20), nullable=False)
        corrected_label = db.Column(db.String(20), nullable=True)
        confidence = db.Column(db.Float, nullable=True)
        user_id = db.Column(db.Integer, nullable=True)
        timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
        is_reviewed = db.Column(db.Boolean, default=False, nullable=False)

        def to_dict(self) -> dict:
            """Serialize to a plain dict for JSON/template rendering."""
            return {
                "id": self.id,
                "platform": self.platform,
                "profile_data": json.loads(self.profile_data_json or "{}"),
                "predicted_label": self.predicted_label,
                "corrected_label": self.corrected_label,
                "confidence": self.confidence,
                "user_id": self.user_id,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                "is_reviewed": self.is_reviewed,
            }

    return FeedbackEntry


# ---------------------------------------------------------------------------
# Helper functions — accept db session and model class as arguments
# ---------------------------------------------------------------------------

def save_feedback(
    db,
    FeedbackModel,
    platform: str,
    profile_data: dict,
    predicted_label: str,
    corrected_label: Optional[str] = None,
    confidence: Optional[float] = None,
    user_id: Optional[int] = None,
) -> int:
    """
    Save a feedback entry to the database.

    Args:
        db: SQLAlchemy db instance.
        FeedbackModel: The FeedbackEntry model class (returned by init_feedback).
        platform: Platform name (e.g., "instagram").
        profile_data: Dict of profile features that were predicted.
        predicted_label: The model's prediction ("Fake" or "Legit").
        corrected_label: User's correction (None if just flagging, "Fake"/"Legit" if correcting).
        confidence: Model confidence score (0-100).
        user_id: ID of the logged-in user (None in no-auth mode).

    Returns:
        ID of the newly created feedback entry.
    """
    try:
        entry = FeedbackModel(
            platform=platform,
            profile_data_json=json.dumps(profile_data),
            predicted_label=predicted_label,
            corrected_label=corrected_label,
            confidence=confidence,
            user_id=user_id,
            is_reviewed=False,
        )
        db.session.add(entry)
        db.session.commit()
        logger.info("Saved feedback entry id=%d platform=%s prediction=%s correction=%s",
                    entry.id, platform, predicted_label, corrected_label)
        return entry.id
    except Exception as exc:
        db.session.rollback()
        logger.error("save_feedback error: %s", exc)
        raise


def get_pending_reviews(db, FeedbackModel, limit: int = 100) -> List[dict]:
    """
    Retrieve unreviewed feedback entries, ordered by most recent first.

    Args:
        db: SQLAlchemy db instance.
        FeedbackModel: The FeedbackEntry model class.
        limit: Maximum number of entries to return.

    Returns:
        List of dicts (from FeedbackEntry.to_dict()).
    """
    try:
        entries = (
            FeedbackModel.query
            .filter_by(is_reviewed=False)
            .order_by(FeedbackModel.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [e.to_dict() for e in entries]
    except Exception as exc:
        logger.error("get_pending_reviews error: %s", exc)
        return []


def get_all_feedback(db, FeedbackModel, limit: int = 500) -> List[dict]:
    """
    Retrieve all feedback entries, ordered by most recent first.

    Args:
        db: SQLAlchemy db instance.
        FeedbackModel: The FeedbackEntry model class.
        limit: Maximum number of entries to return.

    Returns:
        List of dicts.
    """
    try:
        entries = (
            FeedbackModel.query
            .order_by(FeedbackModel.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [e.to_dict() for e in entries]
    except Exception as exc:
        logger.error("get_all_feedback error: %s", exc)
        return []


def get_feedback_for_retraining(db, FeedbackModel, platform: str) -> List[dict]:
    """
    Get all feedback entries that have corrected labels for a given platform.
    These are used to build a corrected training set.

    Args:
        db: SQLAlchemy db instance.
        FeedbackModel: The FeedbackEntry model class.
        platform: Platform name.

    Returns:
        List of dicts where corrected_label is not None.
    """
    try:
        entries = (
            FeedbackModel.query
            .filter(
                FeedbackModel.platform == platform,
                FeedbackModel.corrected_label.isnot(None),
            )
            .all()
        )
        return [e.to_dict() for e in entries]
    except Exception as exc:
        logger.error("get_feedback_for_retraining error: %s", exc)
        return []


def mark_reviewed(db, FeedbackModel, entry_id: int) -> bool:
    """
    Mark a feedback entry as reviewed by an admin.

    Args:
        db: SQLAlchemy db instance.
        FeedbackModel: The FeedbackEntry model class.
        entry_id: ID of the feedback entry.

    Returns:
        True if found and updated, False if not found.
    """
    try:
        entry = FeedbackModel.query.get(entry_id)
        if entry is None:
            return False
        entry.is_reviewed = True
        db.session.commit()
        return True
    except Exception as exc:
        db.session.rollback()
        logger.error("mark_reviewed error: %s", exc)
        return False


def get_feedback_stats(db, FeedbackModel) -> dict:
    """
    Compute aggregate statistics about the feedback dataset.

    Returns:
        Dict with:
          total_feedback    — total number of entries
          pending_reviews   — number of unreviewed entries
          correction_rate   — fraction with corrected_label set
          platforms_dict    — dict: platform -> count of entries
    """
    try:
        total = FeedbackModel.query.count()
        pending = FeedbackModel.query.filter_by(is_reviewed=False).count()
        with_correction = FeedbackModel.query.filter(
            FeedbackModel.corrected_label.isnot(None)
        ).count()

        # Group by platform
        platforms_dict: Dict[str, int] = {}
        entries = FeedbackModel.query.all()
        for e in entries:
            platforms_dict[e.platform] = platforms_dict.get(e.platform, 0) + 1

        return {
            "total_feedback": total,
            "pending_reviews": pending,
            "correction_rate": round(with_correction / max(total, 1), 3),
            "platforms_dict": platforms_dict,
        }
    except Exception as exc:
        logger.error("get_feedback_stats error: %s", exc)
        return {
            "total_feedback": 0,
            "pending_reviews": 0,
            "correction_rate": 0.0,
            "platforms_dict": {},
        }
