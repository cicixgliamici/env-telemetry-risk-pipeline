import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.config import RAW_DATA_PATH, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_csv
from src.preprocessing import clean_dataset, split_features_target
from src.feature_engineering import add_engineered_features
from src.evaluate import evaluate_classifier
from src.visualize import plot_feature_importance, plot_confusion_matrix
from src.utils import ensure_dir, save_json

def run_training() -> dict:
    for path in [PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, FIGURES_DIR]:
        ensure_dir(path)
    df = load_csv(RAW_DATA_PATH)
    cleaned = clean_dataset(df)
    cleaned.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)
    featured = add_engineered_features(cleaned)
    featured.to_csv(PROCESSED_DIR / "featured_dataset.csv", index=False)
    X_train, X_test, y_train, y_test = split_features_target(featured, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    model = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced")
    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)
    save_json(metrics, OUTPUT_DIR / "metrics.json")
    pd.DataFrame({"y_true": y_test.reset_index(drop=True), "y_pred": pd.Series(model.predict(X_test))}).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    joblib.dump(model, MODELS_DIR / "baseline_model.pkl")
    plot_feature_importance(model, X_train.columns, FIGURES_DIR / "feature_importance.png")
    plot_confusion_matrix(metrics["confusion_matrix"], FIGURES_DIR / "confusion_matrix.png")
    return metrics

if __name__ == "__main__":
    print(run_training())
