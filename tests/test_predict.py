from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.predict import prepare_prediction_features, run_batch_prediction
from src.train import run_training


class ConstantModel:
    def predict(self, X):
        return [1] * len(X)


class HumidityModel:
    """Return row-dependent outputs to expose ordering errors in batch inference."""

    def predict(self, X):
        return (X["humidity_pct"] >= 80).astype(int).to_numpy()

    def predict_proba(self, X):
        probability = X["humidity_pct"].to_numpy() / 100.0
        return pd.DataFrame({0: 1 - probability, 1: probability}).to_numpy()


@pytest.mark.parametrize(
    "row_positions",
    [
        [7, 0, 6, 1, 5, 2, 4, 3],
        [0, 1, 1, 2, 3, 4, 5, 6, 7],
        [7, 0, 7, 1, 6, 1, 5, 2, 4, 3],
    ],
    ids=["shuffled", "duplicates", "shuffled_duplicates"],
)
def test_prediction_preserves_input_rows(valid_dataset_df, tmp_path: Path, row_positions):
    dataset = valid_dataset_df.iloc[row_positions].reset_index(drop=True)
    input_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "predictions.csv"
    model_path = tmp_path / "model.pkl"
    dataset.to_csv(input_path, index=False)
    joblib.dump(HumidityModel(), model_path)

    features = prepare_prediction_features(input_path, feature_columns=["humidity_pct"])
    pd.testing.assert_series_equal(features["humidity_pct"], dataset["humidity_pct"])

    result = run_batch_prediction(
        model_path=model_path,
        raw_data_path=input_path,
        output_path=output_path,
        metadata_path=None,
    )

    # Check persisted results too: both labels and probabilities must follow input order.
    for predictions in [result, pd.read_csv(output_path)]:
        assert len(predictions) == len(dataset)
        assert pd.to_datetime(predictions["timestamp"]).tolist() == pd.to_datetime(
            dataset["timestamp"]
        ).tolist()
        pd.testing.assert_frame_equal(
            predictions[dataset.columns].drop(columns="timestamp"),
            dataset.drop(columns="timestamp"),
        )
        assert predictions["predicted_risk_level"].tolist() == (
            dataset["humidity_pct"] >= 80
        ).astype(int).tolist()
        assert predictions["predicted_risk_probability"].tolist() == pytest.approx(
            dataset["humidity_pct"] / 100.0
        )


def test_prepare_prediction_features_uses_metadata_feature_order(valid_dataset_df, tmp_path: Path):
    input_path = tmp_path / "dataset.csv"
    valid_dataset_df.to_csv(input_path, index=False)
    feature_columns = ["humidity_temp_index", "pressure_deviation"]

    X = prepare_prediction_features(input_path, feature_columns=feature_columns)

    assert list(X.columns) == feature_columns
    assert len(X) == len(valid_dataset_df)


def test_run_batch_prediction_writes_output(valid_dataset_df, tmp_path: Path):
    input_path = tmp_path / "dataset.csv"
    output_path = tmp_path / "predictions.csv"
    model_path = tmp_path / "model.pkl"
    valid_dataset_df.to_csv(input_path, index=False)
    joblib.dump(ConstantModel(), model_path)

    result = run_batch_prediction(
        model_path=model_path,
        raw_data_path=input_path,
        output_path=output_path,
        metadata_path=None,
    )

    assert "predicted_risk_level" in result.columns
    assert result["predicted_risk_level"].eq(1).all()
    assert output_path.exists()
    assert "predicted_risk_level" in pd.read_csv(output_path).columns


def test_batch_prediction_can_use_training_metadata():
    metrics = run_training()
    model_path = Path("models") / f"{metrics['selected_model']}_model.pkl"

    result = run_batch_prediction(model_path=model_path)

    assert "predicted_risk_level" in result.columns
    assert len(result) > 0
