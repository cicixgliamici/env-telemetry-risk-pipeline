import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.data_loader import load_csv
from src.feature_engineering import add_engineered_features
from src.preprocessing import clean_dataset
from src.utils import ensure_dir, get_logger, save_json, utc_timestamp
from src.visualize import plot_anomaly_scatter

LOGGER = get_logger("src.anomaly")


def detect_anomalies(contamination: float = 0.03, raw_data_path=RAW_DATA_PATH):
    """Run anomaly detection and save both row-level and summary artifacts."""

    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and less than or equal to 0.5")

    for path in [PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
        ensure_dir(path)

    LOGGER.info("Running anomaly detection on %s", raw_data_path)
    df = add_engineered_features(clean_dataset(load_csv(raw_data_path)))
    X = df.drop(columns=["timestamp", "risk_level", "failure_risk"], errors="ignore")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    labels = model.fit_predict(X)
    scores = model.decision_function(X)

    # Complement the model output with a simple z-score flag for easier inspection.
    zcols = ["temperature_c", "humidity_pct", "vibration_mm_s", "pollution_proxy"]
    z = np.abs((df[zcols] - df[zcols].mean()) / df[zcols].std(ddof=0))

    df["zscore_flag"] = (z > 3).any(axis=1).astype(int)
    df["anomaly_score"] = scores
    df["is_anomaly"] = (labels == -1).astype(int)

    df.to_csv(OUTPUT_DIR / "anomalies.csv", index=False)
    joblib.dump(model, MODELS_DIR / "anomaly_detector.pkl")
    plot_anomaly_scatter(df, FIGURES_DIR / "anomaly_scatter.png")
    save_json(
        {
            "rows": int(len(df)),
            "anomalies": int(df["is_anomaly"].sum()),
            "contamination": contamination,
            "run_timestamp_utc": utc_timestamp(),
        },
        OUTPUT_DIR / "anomaly_summary.json",
    )
    LOGGER.info("Anomaly detection completed with %s anomalies", int(df["is_anomaly"].sum()))
    return df


if __name__ == "__main__":
    out = detect_anomalies()
    print(out["is_anomaly"].value_counts().to_dict())
