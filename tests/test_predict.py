from pathlib import Path

import joblib
import pandas as pd

from src.predict import prepare_prediction_features, run_batch_prediction
from src.train import run_training


class ConstantModel:
    def predict(self, X):
        return [1] * len(X)


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
