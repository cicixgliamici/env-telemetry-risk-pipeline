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


def plot_threshold_selection(curve: pd.DataFrame, policy: dict, output_path: str | Path) -> None:
    """Plot only training-validation evidence, keeping test scores out of threshold selection."""

    finite = curve.dropna(subset=["threshold"])
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(finite["threshold"], finite["precision_risk"], label="Risk precision")
    axes[0].plot(finite["threshold"], finite["recall_risk"], label="Risk recall")
    axes[0].set_ylabel("Training-validation metric")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[1].plot(finite["threshold"], finite["cost_per_observation"], label="Relative error cost")
    # Never-risk differs from threshold 1 when some scores equal exactly 1.
    never_risk = curve[curve["rule"] == "never_risk"].iloc[0]
    axes[1].scatter([1.1], [never_risk["cost_per_observation"]], marker="x", color="black")
    axes[1].set_ylabel("Cost per validation observation")
    axes[1].set_xlabel("Risk probability threshold (inclusive)")
    for axis in axes:
        axis.grid(alpha=0.25)
        cutoff = policy["threshold"] if policy["threshold"] is not None else 1.1
        axis.axvline(cutoff, color="black", linestyle="--", label="Selected cutoff")
    axes[1].set_xticks(
        [0, 0.2, 0.4, 0.6, 0.8, 1, 1.1], ["0", "0.2", "0.4", "0.6", "0.8", "1", "never\nrisk"]
    )
    rule = "never risk" if policy["threshold"] is None else f"threshold {policy['threshold']:.2f}"
    figure.suptitle(f"Training-Only Decision Selection: {rule}")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
