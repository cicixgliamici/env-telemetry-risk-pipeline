"""Check learned preprocessing statistics and serialized inference behavior."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.feature_engineering import add_engineered_features
from src.modeling import build_model_candidates
from src.predict import prepare_prediction_features, run_batch_prediction
from src.train import _cross_validate_candidates


@pytest.fixture
def missing_telemetry(valid_dataset_df):
    dataset = valid_dataset_df.copy()
    dataset.loc[0, "humidity_pct"] = np.nan
    # A completely missing input must keep its position in the fitted schema.
    dataset["pollution_proxy"] = np.nan
    return dataset


def _raw_features(dataset):
    return dataset.drop(columns=["timestamp", "risk_level", "failure_risk"])


@pytest.mark.parametrize("model_name", ["random_forest", "logistic_regression"])
def test_pipeline_preserves_training_statistics(missing_telemetry, model_name):
    features = _raw_features(missing_telemetry)
    pipeline = build_model_candidates()[model_name]
    pipeline.fit(features, missing_telemetry["risk_level"])
    imputer = pipeline.named_steps["imputer"]
    expected_medians = features.median().fillna(0)
    np.testing.assert_allclose(imputer.statistics_, expected_medians)

    incoming = features.iloc[[0, 1]].copy()
    incoming.loc[incoming.index[1], "humidity_pct"] = 1000
    imputed = imputer.transform(incoming)
    assert imputed.iloc[0]["humidity_pct"] == expected_medians["humidity_pct"]
    assert imputed["pollution_proxy"].eq(0).all()
    assert imputed.columns.tolist() == features.columns.tolist()
    engineered = pipeline.named_steps["feature_engineering"].transform(imputed)
    assert engineered.iloc[0]["humidity_temp_index"] == pytest.approx(
        expected_medians["humidity_pct"] * incoming.iloc[0]["temperature_c"] / 100
    )

    if model_name == "logistic_regression":
        training_features = add_engineered_features(imputer.transform(features))
        scaler = pipeline.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, training_features.mean())
        pipeline.predict(incoming)
        np.testing.assert_allclose(scaler.mean_, training_features.mean())
    else:
        assert pipeline.named_steps["scaler"] == "passthrough"
    np.testing.assert_allclose(imputer.statistics_, expected_medians)


def test_cross_validation_fits_imputers_on_fold_training_rows(missing_telemetry, monkeypatch):
    features = _raw_features(missing_telemetry)
    target = missing_telemetry["risk_level"]
    candidates = build_model_candidates()
    observed_rows = []
    observed_scaler_rows = []
    original_fit = SimpleImputer.fit
    original_scaler_fit = StandardScaler.fit

    def fit_and_check_statistics(imputer, fold_features, y=None):
        fitted = original_fit(imputer, fold_features, y)
        observed_rows.append(tuple(fold_features.index))
        expected_medians = features.loc[fold_features.index].median().fillna(0)
        np.testing.assert_allclose(fitted.statistics_, expected_medians)
        return fitted

    def fit_and_check_scaler(scaler, fold_features, y=None):
        raw_fold = features.loc[fold_features.index]
        expected = add_engineered_features(raw_fold.fillna(raw_fold.median().fillna(0)))
        pd.testing.assert_frame_equal(fold_features, expected, check_dtype=False)
        fitted = original_scaler_fit(scaler, fold_features, y)
        observed_scaler_rows.append(tuple(fold_features.index))
        np.testing.assert_allclose(fitted.mean_, expected.mean())
        return fitted

    monkeypatch.setattr(SimpleImputer, "fit", fit_and_check_statistics)
    monkeypatch.setattr(StandardScaler, "fit", fit_and_check_scaler)
    results = _cross_validate_candidates(candidates, features, target)

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    expected_rows = [
        tuple(features.iloc[train].index) for train, _ in splitter.split(features, target)
    ]
    assert observed_rows == expected_rows * len(candidates)
    assert observed_scaler_rows == expected_rows
    assert all(result["folds"] == 3 for result in results.values())
    # Cross-validation must fit clones, leaving the original candidates unfitted.
    assert all(
        not hasattr(model.named_steps["imputer"], "statistics_") for model in candidates.values()
    )


def test_saved_pipeline_predictions_are_batch_independent(missing_telemetry, tmp_path):
    pipeline = build_model_candidates()["logistic_regression"]
    pipeline.fit(_raw_features(missing_telemetry), missing_telemetry["risk_level"])
    model_path = tmp_path / "pipeline.pkl"
    joblib.dump(pipeline, model_path)
    reference_probability = None

    for companion_humidity in [20, 1000]:
        # Repeated rows, missing values, and shuffled timestamps must survive scoring.
        incoming = missing_telemetry.iloc[[7, 0, 0, 4]].reset_index(drop=True)
        incoming.loc[0, "humidity_pct"] = companion_humidity
        input_path = tmp_path / "incoming.csv"
        incoming.to_csv(input_path, index=False)
        prepared = prepare_prediction_features(input_path)
        pd.testing.assert_frame_equal(prepared, _raw_features(incoming))
        result = run_batch_prediction(
            model_path,
            input_path,
            output_path=tmp_path / "predictions.csv",
            metadata_path=tmp_path / "nonexistent_metadata.json",
        )
        expected = pipeline.predict_proba(_raw_features(incoming))[:, 1]
        np.testing.assert_allclose(result["predicted_risk_probability"], expected)
        assert result["predicted_risk_level"].tolist() == pipeline.predict(
            _raw_features(incoming)
        ).tolist()
        pd.testing.assert_series_equal(result["humidity_pct"], incoming["humidity_pct"])
        if reference_probability is None:
            reference_probability = result.loc[1, "predicted_risk_probability"]
        assert result.loc[1, "predicted_risk_probability"] == reference_probability
