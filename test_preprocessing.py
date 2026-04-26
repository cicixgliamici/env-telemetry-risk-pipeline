import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from src.config import RAW_DATA_PATH, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, RANDOM_STATE, PROCESSED_DIR
from src.data_loader import load_csv
from src.preprocessing import clean_dataset
from src.feature_engineering import add_engineered_features
from src.visualize import plot_anomaly_scatter
from src.utils import ensure_dir

def detect_anomalies(contamination: float = 0.03):
    for path in [PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
        ensure_dir(path)
    df = add_engineered_features(clean_dataset(load_csv(RAW_DATA_PATH)))
    X = df.drop(columns=["timestamp", "risk_level", "failure_risk"], errors="ignore")
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    scores = model.decision_function(X)
    zcols = ["temperature_c", "humidity_pct", "vibration_mm_s", "pollution_proxy"]
    z = np.abs((df[zcols] - df[zcols].mean()) / df[zcols].std(ddof=0))
    df["zscore_flag"] = (z > 3).any(axis=1).astype(int)
    df["anomaly_score"] = scores
    df["is_anomaly"] = (labels == -1).astype(int)
    df.to_csv(OUTPUT_DIR / "anomalies.csv", index=False)
    joblib.dump(model, MODELS_DIR / "anomaly_detector.pkl")
    plot_anomaly_scatter(df, FIGURES_DIR / "anomaly_scatter.png")
    return df

if __name__ == "__main__":
    out = detect_anomalies()
    print(out["is_anomaly"].value_counts().to_dict())
