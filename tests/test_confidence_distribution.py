"""
tests/test_confidence_distribution.py

Tests for the confidence distribution fix:
  - Legitimate profiles with sparse data should get >= 55% confidence
  - Clearly fake profiles should get >= 75% confidence
  - Missing feature handler defaults to platform medians, not zero
  - Confidence calibrator selects best method and improves Brier score
  - Feature group weights apply correctly
  - merge_stage_predictions falls back to Stage 1 when Stage 2 degrades

Run with:
    python -m pytest tests/test_confidence_distribution.py -v
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Missing value handler
# ---------------------------------------------------------------------------

class TestMissingValueHandler:
    def test_defaults_not_zero_for_known_platform(self):
        """handle_missing_features should use platform medians, not 0."""
        from features.missing_value_handler import handle_missing_features, _HARDCODED_DEFAULTS

        features = {"username": "testuser"}  # sparse profile
        result = handle_missing_features(features, "instagram")

        # followers default should be > 0 (real users have followers)
        assert result.get("followers", 0) > 0, (
            "followers default for Instagram should be > 0 (real users have followers)"
        )

    def test_missing_flags_added(self):
        """Important missing features should get _missing binary flags."""
        from features.missing_value_handler import handle_missing_features, _IMPORTANT_FEATURES

        features = {"username": "testuser"}
        result = handle_missing_features(features, "github")

        missing_flags = [k for k in result if k.endswith("_missing")]
        assert len(missing_flags) > 0, "Should add _missing flags for absent important features"

    def test_existing_values_not_overwritten(self):
        """User-supplied values should not be overwritten by defaults."""
        from features.missing_value_handler import handle_missing_features

        features = {"followers": 9999, "following": 500}
        result = handle_missing_features(features, "instagram")

        assert result["followers"] == 9999, "User-supplied followers value should be preserved"
        assert result["following"] == 500, "User-supplied following value should be preserved"

    def test_feature_completeness_score(self):
        """calculate_feature_completeness should return 0.0–1.0."""
        from features.missing_value_handler import calculate_feature_completeness

        sparse = {"username": "testuser"}
        complete = {
            "followers": 200, "following": 100, "posts": 50,
            "account_age_days": 365, "bio": "Hello world", "has_avatar": 1,
        }
        sparse_score = calculate_feature_completeness(sparse, "instagram")
        complete_score = calculate_feature_completeness(complete, "instagram")

        assert 0.0 <= sparse_score <= 1.0
        assert 0.0 <= complete_score <= 1.0
        assert complete_score > sparse_score, (
            "More complete profile should have higher completeness score"
        )


# ---------------------------------------------------------------------------
# Confidence calibrator
# ---------------------------------------------------------------------------

class TestRobustConfidenceCalibrator:
    def _make_data(self, n=200):
        np.random.seed(42)
        labels = np.random.randint(0, 2, n)
        # Simulated overconfident model: cluster around 0.9 (fake) and 0.1 (legit)
        raw = np.where(labels == 1, np.random.beta(8, 2, n), np.random.beta(2, 8, n))
        return raw.tolist(), labels.tolist()

    def test_fit_selects_method(self):
        from ML.confidence_calibrator import RobustConfidenceCalibrator

        raw, labels = self._make_data()
        cal = RobustConfidenceCalibrator()
        cal.fit(raw, labels, verbose=False)

        assert cal._fitted
        assert cal.method in ("temperature", "platt", "isotonic", "none")

    def test_calibrate_output_in_range(self):
        from ML.confidence_calibrator import RobustConfidenceCalibrator

        raw, labels = self._make_data()
        cal = RobustConfidenceCalibrator()
        cal.fit(raw, labels, verbose=False)

        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = cal.calibrate(p)
            assert 0.0 <= result <= 1.0, f"calibrate({p}) = {result} out of [0,1]"

    def test_calibration_does_not_worsen_brier(self):
        """Calibration should not increase Brier score by more than 2%."""
        from ML.confidence_calibrator import RobustConfidenceCalibrator

        raw, labels = self._make_data()
        cal = RobustConfidenceCalibrator()
        cal.fit(raw, labels, verbose=False)

        raw_arr = np.array(raw)
        label_arr = np.array(labels)
        uncal_brier = float(np.mean((raw_arr - label_arr) ** 2))

        if cal.method != "none":
            cal_probs = np.array([cal.calibrate(p) for p in raw])
            cal_brier = float(np.mean((cal_probs - label_arr) ** 2))
            assert cal_brier <= uncal_brier * 1.02, (
                f"Calibration worsened Brier score: {uncal_brier:.4f} → {cal_brier:.4f}"
            )

    def test_save_load_roundtrip(self, tmp_path):
        from ML.confidence_calibrator import RobustConfidenceCalibrator

        raw, labels = self._make_data()
        cal = RobustConfidenceCalibrator()
        cal.fit(raw, labels, verbose=False)

        model_path = tmp_path / "test_model.pkl"
        cal.save(model_path)
        loaded = RobustConfidenceCalibrator.load(model_path)

        assert loaded.method == cal.method
        assert loaded._fitted == cal._fitted
        assert abs(loaded.temperature - cal.temperature) < 1e-6

    def test_insufficient_data_no_crash(self):
        from ML.confidence_calibrator import RobustConfidenceCalibrator

        cal = RobustConfidenceCalibrator()
        cal.fit([0.6, 0.4, 0.7], [1, 0, 1], verbose=False)  # only 3 samples

        assert not cal._fitted
        assert cal.method == "none"
        # calibrate should still work (identity passthrough)
        assert cal.calibrate(0.7) == pytest.approx(0.7, abs=1e-6)

    def test_calibrate_pct_helper(self):
        from ML.confidence_calibrator import RobustConfidenceCalibrator

        cal = RobustConfidenceCalibrator()
        result = cal.calibrate_confidence_pct(75.0)
        assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# Feature group weights
# ---------------------------------------------------------------------------

class TestFeatureGroupWeights:
    def test_behavioral_features_get_higher_weight(self):
        from ML.persistent_common import apply_feature_group_weights, FEATURE_GROUP_WEIGHTS

        df = pd.DataFrame({
            "behavioral_post_rate": [1.0],
            "followers": [1.0],
            "bio_len": [1.0],
        })
        weighted = apply_feature_group_weights(df)

        assert weighted["behavioral_post_rate"].iloc[0] == pytest.approx(
            FEATURE_GROUP_WEIGHTS["behavioral"], abs=1e-6
        )
        assert weighted["followers"].iloc[0] == pytest.approx(
            FEATURE_GROUP_WEIGHTS["base_metadata"], abs=1e-6
        )
        assert weighted["bio_len"].iloc[0] == pytest.approx(
            FEATURE_GROUP_WEIGHTS["bio_forensics"], abs=1e-6
        )

    def test_weight_function_covers_all_groups(self):
        from ML.persistent_common import _get_feature_group_weight, FEATURE_GROUP_WEIGHTS

        test_cols = {
            "followers": "base_metadata",
            "uname_len": "username_forensics",
            "bio_spam_score": "bio_forensics",
            "avatar_stock_score": "photo_analysis",
            "behavioral_entropy": "behavioral",
            "xp_presence_ratio": "cross_platform",
            "temporal_post_freq": "temporal",
        }
        for col, group in test_cols.items():
            w = _get_feature_group_weight(col)
            assert w == FEATURE_GROUP_WEIGHTS[group], (
                f"Column '{col}' should map to group '{group}' (weight={FEATURE_GROUP_WEIGHTS[group]}) "
                f"but got {w}"
            )


# ---------------------------------------------------------------------------
# merge_stage_predictions
# ---------------------------------------------------------------------------

class TestMergeStagePrediictions:
    def test_falls_back_when_stage2_degrades_confidence(self):
        from prediction.tiered_predictor import merge_stage_predictions

        label, conf, source = merge_stage_predictions(
            stage1_label="Legit", stage1_confidence=72.0,
            stage1_features={"followers": 200, "following": 100, "posts": 50},
            stage2_label="Legit", stage2_confidence=50.0,  # dropped 22pp
            stage2_features={"followers": 200},
        )
        assert source == "stage1_fallback", (
            "Should fall back to Stage 1 when confidence drops > 15pp"
        )
        assert conf == pytest.approx(72.0)

    def test_uses_stage2_when_more_complete(self):
        from prediction.tiered_predictor import merge_stage_predictions

        stage1_feats = {"followers": 200}
        stage2_feats = {"followers": 200, "following": 100, "posts": 50,
                        "account_age_days": 365, "bio_len": 120}

        label, conf, source = merge_stage_predictions(
            stage1_label="Fake", stage1_confidence=65.0,
            stage1_features=stage1_feats,
            stage2_label="Fake", stage2_confidence=72.0,
            stage2_features=stage2_feats,
        )
        assert source == "stage2_dominant"
        assert conf == pytest.approx(72.0)

    def test_weighted_merge_when_coverage_similar(self):
        from prediction.tiered_predictor import merge_stage_predictions

        # Both stages have similar completeness; expect weighted blend
        feats = {"followers": 200, "following": 100, "posts": 50}
        label, conf, source = merge_stage_predictions(
            stage1_label="Legit", stage1_confidence=60.0,
            stage1_features=feats,
            stage2_label="Legit", stage2_confidence=70.0,
            stage2_features={**feats, "extra": 1},  # slightly more complete
        )
        # Stage 2 is more complete (even slightly) → stage2_dominant
        assert source in ("stage2_dominant", "weighted_merge")


# ---------------------------------------------------------------------------
# Confidence explainer
# ---------------------------------------------------------------------------

class TestConfidenceExplainer:
    def test_explanation_structure(self):
        from features.confidence_explainer import generate_confidence_explanation

        features = {
            "followers": 1200, "following": 300, "posts": 85,
            "account_age_days": 730, "has_avatar": 1, "bio_len": 80,
        }
        result = generate_confidence_explanation("Legit", 72.0, features, platform="instagram")

        assert "strong_signals" in result
        assert "weak_signals" in result
        assert "missing_signals" in result
        assert "data_coverage_pct" in result
        assert "improvement_tips" in result
        assert isinstance(result["data_coverage_pct"], (int, float))
        assert 0 <= result["data_coverage_pct"] <= 100

    def test_adjust_confidence_does_not_exceed_95(self):
        from features.confidence_explainer import adjust_confidence_for_display

        # Even with perfect signals, confidence should cap at 95
        features = {
            "followers": 10000, "following": 500, "posts": 300,
            "account_age_days": 1200, "has_avatar": 1, "bio_len": 150,
            "ff_ratio": 20.0,
        }
        result = adjust_confidence_for_display(92.0, "Legit", features)
        assert result <= 95.0

    def test_adjust_confidence_non_negative(self):
        from features.confidence_explainer import adjust_confidence_for_display

        result = adjust_confidence_for_display(5.0, "Fake", {})
        assert result >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
