import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.experiment import build_data_evidence, build_feature_ablation
from src.modeling import build_model_candidates
from src.train import _cross_validate_candidates


def test_feature_ablation_reports_positive_and_negative_changes():
    comparison = {
        "random_forest_raw": {
            "f1_weighted_mean": 0.6,
            "f1_macro_mean": 0.5,
            "recall_risk_mean": 0.8,
        },
        "random_forest": {"f1_weighted_mean": 0.5, "f1_macro_mean": 0.65, "recall_risk_mean": 0.7},
    }

    ablation = build_feature_ablation(comparison)["random_forest"]

    assert ablation["f1_weighted_delta"] == pytest.approx(-0.1)
    assert ablation["f1_macro_delta"] == pytest.approx(0.15)
    assert ablation["recall_risk_delta"] == pytest.approx(-0.1)


def test_dataset_evidence_is_reproducible_without_claiming_verified_provenance(valid_dataset_df):
    original = build_data_evidence(valid_dataset_df)
    assert original == build_data_evidence(valid_dataset_df.copy())
    valid_dataset_df.loc[0, "humidity_pct"] += 1
    changed = build_data_evidence(valid_dataset_df)

    assert original["validated_csv_sha256"] != changed["validated_csv_sha256"]
    assert original["provenance_status"] == "not externally verified"
    assert original["rows"] == 8


def test_raw_and_engineered_candidates_preserve_family_settings(valid_dataset_df):
    candidates = build_model_candidates()
    features = valid_dataset_df.drop(columns=["timestamp", "risk_level", "failure_risk"])
    for family in ["random_forest", "logistic_regression"]:
        engineered, raw = candidates[family], candidates[f"{family}_raw"]
        assert (
            engineered.named_steps["classifier"].get_params()
            == raw.named_steps["classifier"].get_params()
        )
        assert engineered.named_steps["classifier"] is not raw.named_steps["classifier"]
        raw.fit(features, valid_dataset_df["risk_level"])
        engineered.fit(features, valid_dataset_df["risk_level"])
        assert raw[:2].transform(features).shape[1] == 8
        assert engineered[:2].transform(features).shape[1] == 16


def test_dummy_baseline_ignores_feature_values(valid_dataset_df):
    dummy = build_model_candidates()["dummy_most_frequent"]
    assert isinstance(dummy.named_steps["classifier"], DummyClassifier)
    features = valid_dataset_df.drop(columns=["timestamp", "risk_level", "failure_risk"])
    # Use an actual majority rather than the balanced fixture's arbitrary tie.
    labels = [0, 0, 0, 0, 0, 0, 1, 1]
    dummy.fit(features, labels)

    assert dummy.predict(features).tolist() == [0] * 8
    np.testing.assert_array_equal(dummy.predict(features * 2), dummy.predict(features))


def test_cross_validation_marks_risk_recall_unavailable_for_other_class_codes(valid_dataset_df):
    features = valid_dataset_df.drop(columns=["timestamp", "risk_level", "failure_risk"])
    target = valid_dataset_df["risk_level"] + 2
    dummy = build_model_candidates()["dummy_most_frequent"]

    result = _cross_validate_candidates({"dummy_most_frequent": dummy}, features, target)

    assert result["dummy_most_frequent"]["recall_risk_mean"] is None
    assert result["dummy_most_frequent"]["recall_risk_std"] is None
    assert result["dummy_most_frequent"]["f1_macro_mean"] >= 0
