"""
ML/confidence_calibrator.py — Robust multi-method confidence calibration.

Fits three calibrators and selects the one with the best Brier score:
  1. Isotonic regression (current method — good for large datasets)
  2. Platt scaling (logistic regression on log-odds — stable for small data)
  3. Temperature scaling (single parameter — most stable, low variance)

Also supports re-calibration from user feedback (FeedbackEntry table).

Usage:
    calibrator = RobustConfidenceCalibrator()
    calibrator.fit(raw_probas, true_labels)
    adjusted = calibrator.calibrate(0.62)

    # Save/load alongside the model
    calibrator.save(path)
    calibrator = RobustConfidenceCalibrator.load(path)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_CALIBRATOR_SUFFIX = "_calibrator.json"
_MIN_SAMPLES_FOR_CALIBRATION = 20


class RobustConfidenceCalibrator:
    """
    Multi-method confidence calibrator that selects the best fit.

    Attributes:
        method: The calibration method selected ('isotonic', 'platt', 'temperature', 'none')
        temperature: Temperature scaling parameter (applies to 'temperature' method)
        platt_a, platt_b: Platt scaling parameters
        isotonic_bins: Isotonic regression lookup table
    """

    def __init__(self):
        self.method = "none"
        self.temperature = 1.0
        self.platt_a = 1.0
        self.platt_b = 0.0
        self.isotonic_bins: List = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Calibration methods
    # ------------------------------------------------------------------

    @staticmethod
    def _brier_score(probas: np.ndarray, labels: np.ndarray) -> float:
        """Brier score (lower = better)."""
        return float(np.mean((probas - labels) ** 2))

    @staticmethod
    def _apply_temperature(raw_proba: float, T: float) -> float:
        """Apply temperature scaling to a probability."""
        if T <= 0:
            return raw_proba
        # Work in log-odds space
        p = float(np.clip(raw_proba, 1e-7, 1 - 1e-7))
        logit = math.log(p / (1 - p))
        scaled_logit = logit / T
        return 1.0 / (1.0 + math.exp(-scaled_logit))

    @staticmethod
    def _apply_platt(raw_proba: float, a: float, b: float) -> float:
        """Apply Platt scaling: sigmoid(a * logit(p) + b)."""
        p = float(np.clip(raw_proba, 1e-7, 1 - 1e-7))
        logit = math.log(p / (1 - p))
        return 1.0 / (1.0 + math.exp(-(a * logit + b)))

    @staticmethod
    def _fit_isotonic(probas: np.ndarray, labels: np.ndarray) -> List:
        """Fit isotonic regression and return as a lookup table."""
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probas, labels)
            # Store as quantile lookup (100 bins)
            bins = []
            for q in np.linspace(0, 1, 101):
                raw = float(q)
                cal = float(np.clip(iso.predict([raw])[0], 0, 1))
                bins.append((raw, cal))
            return bins
        except Exception as exc:
            logger.debug("Isotonic fit failed: %s", exc)
            return []

    def _apply_isotonic(self, raw_proba: float) -> float:
        """Lookup calibrated probability from isotonic bins."""
        if not self.isotonic_bins:
            return raw_proba
        # Linear interpolation between nearest bins
        p = float(np.clip(raw_proba, 0, 1))
        for i in range(len(self.isotonic_bins) - 1):
            lo_raw, lo_cal = self.isotonic_bins[i]
            hi_raw, hi_cal = self.isotonic_bins[i + 1]
            if lo_raw <= p <= hi_raw:
                if hi_raw == lo_raw:
                    return lo_cal
                t = (p - lo_raw) / (hi_raw - lo_raw)
                return lo_cal + t * (hi_cal - lo_cal)
        return self.isotonic_bins[-1][1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        raw_probas: List[float],
        true_labels: List[int],
        verbose: bool = True,
    ) -> "RobustConfidenceCalibrator":
        """
        Fit all three calibrators and select the best by Brier score.

        Args:
            raw_probas: Raw model probabilities for class=Fake (0-1).
            true_labels: Ground-truth labels (1=Fake, 0=Legit).

        Returns:
            self
        """
        probas = np.array(raw_probas, dtype=float)
        labels = np.array(true_labels, dtype=float)

        if len(probas) < _MIN_SAMPLES_FOR_CALIBRATION:
            logger.warning(
                "Only %d samples for calibration (min %d) — skipping",
                len(probas), _MIN_SAMPLES_FOR_CALIBRATION,
            )
            self.method = "none"
            self._fitted = False
            return self

        # ----- 1. Temperature scaling -----
        best_T = 1.0
        best_T_brier = float("inf")
        try:
            for T in np.arange(0.3, 3.0, 0.1):
                cal = np.array([self._apply_temperature(p, T) for p in probas])
                bs = self._brier_score(cal, labels)
                if bs < best_T_brier:
                    best_T_brier = bs
                    best_T = T
        except Exception as exc:
            logger.debug("Temperature scaling fit failed: %s", exc)
            best_T_brier = float("inf")

        # ----- 2. Platt scaling -----
        best_a, best_b = 1.0, 0.0
        best_platt_brier = float("inf")
        try:
            from scipy.optimize import minimize

            def platt_loss(params):
                a, b = params
                cal = np.array([self._apply_platt(p, a, b) for p in probas])
                cal = np.clip(cal, 1e-7, 1 - 1e-7)
                return -np.mean(labels * np.log(cal) + (1 - labels) * np.log(1 - cal))

            result = minimize(platt_loss, [1.0, 0.0], method="Nelder-Mead",
                              options={"maxiter": 500, "xatol": 1e-4})
            best_a, best_b = result.x
            cal = np.array([self._apply_platt(p, best_a, best_b) for p in probas])
            best_platt_brier = self._brier_score(cal, labels)
        except Exception:
            # Fallback: skip scipy, simple grid search
            best_platt_brier = float("inf")

        # ----- 3. Isotonic regression -----
        iso_bins = self._fit_isotonic(probas, labels)
        best_iso_brier = float("inf")
        if iso_bins:
            self.isotonic_bins = iso_bins
            cal_iso = np.array([self._apply_isotonic(p) for p in probas])
            best_iso_brier = self._brier_score(cal_iso, labels)

        # ----- Select best -----
        uncalibrated_brier = self._brier_score(probas, labels)
        scores = {
            "temperature": best_T_brier,
            "platt": best_platt_brier,
            "isotonic": best_iso_brier,
            "none": uncalibrated_brier,
        }
        best_method = min(scores, key=scores.get)

        # Only use calibration if it actually improves Brier score
        if scores[best_method] >= uncalibrated_brier * 0.98:
            best_method = "none"

        self.method = best_method
        self.temperature = best_T
        self.platt_a = best_a
        self.platt_b = best_b
        self._fitted = True

        if verbose:
            logger.info(
                "Calibration: selected method=%s brier_scores=%s (uncalibrated=%.4f)",
                best_method,
                {k: f"{v:.4f}" for k, v in scores.items()},
                uncalibrated_brier,
            )

        return self

    def calibrate(self, raw_proba: float) -> float:
        """
        Apply the selected calibration method to a single probability.

        Args:
            raw_proba: Raw model probability (0-1).

        Returns:
            Calibrated probability (0-1).
        """
        if not self._fitted or self.method == "none":
            return float(np.clip(raw_proba, 0, 1))

        try:
            if self.method == "temperature":
                return float(np.clip(self._apply_temperature(raw_proba, self.temperature), 0, 1))
            elif self.method == "platt":
                return float(np.clip(self._apply_platt(raw_proba, self.platt_a, self.platt_b), 0, 1))
            elif self.method == "isotonic":
                return float(np.clip(self._apply_isotonic(raw_proba), 0, 1))
        except Exception as exc:
            logger.debug("Calibration apply failed: %s — returning raw", exc)

        return float(np.clip(raw_proba, 0, 1))

    def calibrate_confidence_pct(self, raw_confidence_pct: float) -> float:
        """Calibrate a confidence percentage (0-100). Returns 0-100."""
        return self.calibrate(raw_confidence_pct / 100.0) * 100.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: "Path") -> Path:
        """Save calibrator alongside a model file."""
        cal_path = Path(str(model_path).replace(".pkl", _CALIBRATOR_SUFFIX))
        state = {
            "method": self.method,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "isotonic_bins": self.isotonic_bins,
            "fitted": self._fitted,
        }
        cal_path.write_text(json.dumps(state, indent=2))
        logger.debug("Saved calibrator to %s (method=%s)", cal_path, self.method)
        return cal_path

    @classmethod
    def load(cls, model_path: "Path") -> "RobustConfidenceCalibrator":
        """Load calibrator from alongside a model file. Returns uncalibrated instance on failure."""
        cal_path = Path(str(model_path).replace(".pkl", _CALIBRATOR_SUFFIX))
        obj = cls()
        if not cal_path.exists():
            logger.debug("No calibrator file found at %s — using identity", cal_path)
            return obj
        try:
            state = json.loads(cal_path.read_text())
            obj.method = state.get("method", "none")
            obj.temperature = float(state.get("temperature", 1.0))
            obj.platt_a = float(state.get("platt_a", 1.0))
            obj.platt_b = float(state.get("platt_b", 0.0))
            obj.isotonic_bins = state.get("isotonic_bins", [])
            obj._fitted = state.get("fitted", False)
            logger.debug("Loaded calibrator from %s (method=%s)", cal_path, obj.method)
        except Exception as exc:
            logger.warning("Failed to load calibrator from %s: %s", cal_path, exc)
        return obj


def recalibrate_from_feedback(
    model_path: "Path",
    platform: str,
    min_samples: int = _MIN_SAMPLES_FOR_CALIBRATION,
) -> Optional["RobustConfidenceCalibrator"]:
    """
    Re-calibrate using corrected predictions from the FeedbackEntry table.

    Args:
        model_path: Path to the model file (calibrator saved alongside).
        platform: Platform to pull feedback for.
        min_samples: Minimum feedback entries needed.

    Returns:
        Fitted calibrator or None if insufficient data.
    """
    try:
        from app import FeedbackEntry, db
        entries = FeedbackEntry.query.filter_by(platform=platform).filter(
            FeedbackEntry.corrected_label.isnot(None)
        ).all()

        if len(entries) < min_samples:
            logger.info(
                "Only %d feedback entries for platform=%s (need %d) — skipping recalibration",
                len(entries), platform, min_samples,
            )
            return None

        import json as _json
        raw_probas = []
        true_labels = []
        label_map = {"fake": 1, "Fake": 1, "1": 1, "legit": 0, "Legit": 0, "0": 0}

        for e in entries:
            conf = e.confidence
            corrected = e.corrected_label
            if conf is None or corrected not in label_map:
                continue
            raw_probas.append(conf / 100.0)
            true_labels.append(label_map[corrected])

        if len(raw_probas) < min_samples:
            return None

        calibrator = RobustConfidenceCalibrator()
        calibrator.fit(raw_probas, true_labels)
        calibrator.save(model_path)
        logger.info(
            "Recalibrated from %d feedback entries for platform=%s — method=%s",
            len(raw_probas), platform, calibrator.method,
        )
        return calibrator

    except Exception as exc:
        logger.warning("recalibrate_from_feedback failed for %s: %s", platform, exc)
        return None
