"""Describe model comparisons and dataset identity without overstating the evidence."""

import hashlib
import platform
from importlib.metadata import version

import pandas as pd

from src.config import RISK_LABEL, TARGET_COLUMN


def build_data_evidence(dataset: pd.DataFrame) -> dict:
    """Fingerprint the validated table actually used, independent of its filename."""

    # This identifies a normalized table, not raw file bytes or a verified data source.
    content = dataset.to_csv(index=False).encode("utf-8")
    return {
        "validated_csv_sha256": hashlib.sha256(content).hexdigest(),
        "rows": len(dataset),
        "class_counts": {
            str(label): int(count) for label, count in dataset[TARGET_COLUMN].value_counts().items()
        },
        "provenance_status": "not externally verified",
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package) for package in ["numpy", "pandas", "scikit-learn"]
        },
    }


def build_feature_ablation(comparison: dict) -> dict:
    """Compare each family on identical folds; a positive delta favors engineering."""

    ablation = {}
    for family in ["random_forest", "logistic_regression"]:
        raw_name = f"{family}_raw"
        if family not in comparison or raw_name not in comparison:
            continue
        deltas = {}
        for metric in ["f1_weighted", "f1_macro", "recall_risk"]:
            raw = comparison[raw_name].get(f"{metric}_mean")
            engineered = comparison[family].get(f"{metric}_mean")
            deltas[f"{metric}_delta"] = (
                None if raw is None or engineered is None else engineered - raw
            )
        ablation[family] = {"raw_candidate": raw_name, "engineered_candidate": family, **deltas}
    return ablation


def _score(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _comparison_lines(comparison: dict) -> list[str]:
    lines = [
        "## Candidate Models: Training Cross-Validation",
        "",
        "| Candidate | Folds | Weighted F1 mean +/- std | Macro F1 mean | Risk recall mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in comparison.items():
        lines.append(
            f"| {name} | {values['folds']} | {_score(values['f1_weighted_mean'])} "
            f"+/- {_score(values['f1_weighted_std'])} | {_score(values.get('f1_macro_mean'))} "
            f"| {_score(values.get('recall_risk_mean'))} |"
        )
    return lines


def _ablation_lines(ablation: dict) -> list[str]:
    lines = [
        "## Feature Ablation: Engineered Minus Raw",
        "",
        "| Family | Weighted F1 delta | Macro F1 delta | Risk recall delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family, values in ablation.items():
        lines.append(
            f"| {family} | {_score(values['f1_weighted_delta'])} "
            f"| {_score(values['f1_macro_delta'])} | {_score(values['recall_risk_delta'])} |"
        )
    lines.extend(
        ["", "Deltas use identical training folds and are descriptive, not significance tests."]
    )
    return lines


def format_training_summary(metrics: dict, metadata: dict) -> str:
    """Keep test results, selection evidence, and provenance visibly distinct."""

    lines = [
        "# Training Run Summary",
        "",
        f"- Timestamp (UTC): {metrics['run_timestamp_utc']}",
        f"- Run ID: {metrics['run_id']}",
        f"- Artifacts: {metrics['run_dir']}",
        f"- Selected model: {metrics['selected_model']}",
        "- Selection: highest mean weighted F1 in training-only cross-validation",
        f"- Held-out test accuracy: {_score(metrics['accuracy'])}",
        f"- Held-out test weighted F1: {_score(metrics['f1_weighted'])}",
        f"- Held-out test macro F1: {_score(metrics['f1_macro'])}",
        f"- Held-out test risk recall (class {RISK_LABEL}): {_score(metrics['recall_risk'])}",
        f"- Test risk support: {metrics['risk_support']}",
        f"- Training rows: {metadata['training_rows']}; test rows: {metadata['test_rows']}",
        f"- Selected CV weighted F1 minus dummy: {_score(metrics['dummy_f1_weighted_gap'])}",
        "",
        *_comparison_lines(metrics["cross_validation"]),
        "",
        *_ablation_lines(metrics["feature_ablation"]),
        "",
        "## Held-Out Test Per-Class Metrics",
        "",
        "```text",
        metrics["classification_report"].rstrip(),
        "```",
        "",
        "## Dataset Identity and Limits",
        "",
        f"- Validated CSV SHA-256: {metadata['data_evidence']['validated_csv_sha256']}",
        "- Data origin is not externally verified by the pipeline.",
        "- Tiny or synthetic datasets do not establish real-world predictive performance.",
    ]
    return "\n".join(lines) + "\n"
