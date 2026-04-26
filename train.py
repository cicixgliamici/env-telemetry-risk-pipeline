from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
def plot_feature_importance(model, feature_names, output_path: str | Path) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    s = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    s.plot(kind="bar")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
def plot_confusion_matrix(cm, output_path: str | Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
def plot_anomaly_scatter(df: pd.DataFrame, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 5))
    normal = df[df["is_anomaly"] == 0]
    anomaly = df[df["is_anomaly"] == 1]
    plt.scatter(normal["humidity_pct"], normal["temperature_c"], s=12, alpha=0.5, label="normal")
    plt.scatter(anomaly["humidity_pct"], anomaly["temperature_c"], s=18, alpha=0.8, label="anomaly")
    plt.xlabel("Humidity (%)")
    plt.ylabel("Temperature (C)")
    plt.title("Anomaly Detection Scatter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
