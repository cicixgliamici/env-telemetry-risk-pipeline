import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

# Configure writable Matplotlib storage before its first import in installed environments.
import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
# Select the headless backend before pyplot initializes it.
import matplotlib.pyplot as plt  # noqa: E402


def plot_feature_importance(model, feature_names, output_path: str | Path) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    series = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
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
    plt.scatter(
        normal["humidity_pct"],
        normal["temperature_c"],
        s=12,
        alpha=0.5,
        label="normal",
    )
    plt.scatter(
        anomaly["humidity_pct"],
        anomaly["temperature_c"],
        s=18,
        alpha=0.8,
        label="anomaly",
    )
    plt.xlabel("Humidity (%)")
    plt.ylabel("Temperature (C)")
    plt.title("Anomaly Detection Scatter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_robustness_scores(scenarios: dict, output_path: str | Path) -> None:
    """Show frozen-model F1 sensitivity with descriptive repetition standard deviations."""

    completed = {
        name: result for name, result in scenarios.items() if result["status"] == "completed"
    }
    means = [result["metrics"]["f1_weighted"]["mean"] for result in completed.values()]
    deviations = [result["metrics"]["f1_weighted"]["std"] for result in completed.values()]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.errorbar(means, range(len(completed)), xerr=deviations, fmt="o", capsize=4)
    axis.set_yticks(range(len(completed)), list(completed))
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Weighted F1 (mean +/- repetition std; not a confidence interval)")
    axis.set_title("Frozen Model: Measurement Stress and Case-Mix Sensitivity")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
