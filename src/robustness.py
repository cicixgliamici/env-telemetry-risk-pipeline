"""Measure a frozen classifier's sensitivity to predetermined held-out stress scenarios."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import RunArtifacts, artifact_run
from src.config import RANDOM_STATE, RISK_LABEL, RUNS_DIR, TARGET_COLUMN
from src.evaluate import evaluate_predictions
from src.robustness_reference import EvaluationReference, load_evaluation_reference
from src.stress_scenarios import SCENARIOS, scenario_cases, training_stress_profile
from src.utils import save_json, write_text_file
from src.visualize import plot_robustness_scores

METRICS = ("accuracy", "f1_weighted", "f1_macro", "recall_risk")


def _summary_statistics(values) -> dict:
    observed = [value for value in values if value is not None]
    return {
        "mean": float(np.mean(observed)) if observed else None,
        "std": float(np.std(observed)) if observed else None,
        "observed_repetitions": len(observed),
    }


def _summarize_scenario(records, baseline) -> dict:
    """Negative metric deltas indicate lower scores relative to the original holdout."""

    summary = {"repetitions": len(records), "metrics": {}, "delta_from_baseline": {}}
    for metric in METRICS:
        values = [record["metrics"][metric] for record in records]
        summary["metrics"][metric] = _summary_statistics(values)
        deltas = [
            None if value is None or baseline[metric] is None else value - baseline[metric]
            for value in values
        ]
        summary["delta_from_baseline"][metric] = _summary_statistics(deltas)
    for diagnostic in [
        "prediction_change_rate",
        "missing_fraction",
        "risk_fraction",
        "high_pollution_fraction",
    ]:
        summary[diagnostic] = _summary_statistics(record[diagnostic] for record in records)
    return summary


def _evaluate_scenario(reference, scenario, profile, seed, repetition):
    """Score changed measurements with original labels; pair duplicates by source position."""

    original_features = reference.test[reference.feature_columns]
    features, positions = scenario_cases(original_features, profile, scenario, seed, repetition)
    source_rows = reference.test.iloc[positions]
    target = source_rows[TARGET_COLUMN].to_numpy()
    predictions = (
        reference.baseline_predictions
        if scenario.kind == "baseline"
        else reference.model.predict(features)
    )
    threshold = profile["pollution_upper_quartile"]
    record = {
        "scenario": scenario.name,
        "repetition": repetition,
        "metrics": evaluate_predictions(target, predictions),
        "prediction_change_rate": float(
            np.mean(predictions != reference.baseline_predictions[positions])
        ),
        "missing_fraction": float(features.isna().to_numpy().mean()),
        "risk_fraction": float(np.mean(target == RISK_LABEL)),
        "high_pollution_fraction": float(features["pollution_proxy"].gt(threshold).mean())
        if threshold is not None
        else None,
    }
    audit = features.copy()
    audit.insert(0, "timestamp", source_rows["timestamp"].to_numpy())
    audit.insert(0, "source_row_index", source_rows.index.to_numpy())
    audit.insert(0, "case_position", np.arange(len(features)))
    audit.insert(0, "repetition", repetition)
    audit.insert(0, "scenario", scenario.name)
    audit["y_true"], audit["y_pred"] = target, predictions
    return record, audit


def _run_scenarios(reference: EvaluationReference, artifacts: RunArtifacts, seed, repeats) -> dict:
    profile = training_stress_profile(reference.training[reference.feature_columns])
    save_json(profile, artifacts.outputs / "stress_profile.json")
    baseline = evaluate_predictions(reference.test[TARGET_COLUMN], reference.baseline_predictions)
    summaries, all_records, audit_frames = {}, [], []
    for scenario in SCENARIOS:
        if scenario.kind == "case_mix" and profile["pollution_upper_quartile"] is None:
            # Other stresses remain informative when this sampling rule cannot be defined.
            summaries[scenario.name] = {
                **scenario.settings(),
                "status": "unavailable",
                "repetitions": 0,
                "reason": "No observed training pollution values to define the sampling threshold.",
            }
            continue
        # Baseline and systematic bias are deterministic; repeats add no evidence for them.
        count = 1 if scenario.kind in {"baseline", "bias"} else repeats
        records = []
        for repetition in range(count):
            record, audit = _evaluate_scenario(reference, scenario, profile, seed, repetition)
            records.append(record)
            audit_frames.append(audit)
        all_records.extend(records)
        summaries[scenario.name] = {
            **scenario.settings(),
            "status": "completed",
            **_summarize_scenario(records, baseline),
        }
    pd.concat(audit_frames, ignore_index=True).to_csv(
        artifacts.outputs / "robustness_predictions.csv", index=False
    )
    save_json({"results": all_records}, artifacts.outputs / "robustness_repetitions.json")
    return {
        **artifacts.identity(),
        **reference.identity,
        "seed": seed,
        "repeats": repeats,
        "test_rows": len(reference.test),
        "baseline_metrics": baseline,
        "scenarios": summaries,
    }


def _format_score(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def format_robustness_summary(result: dict) -> str:
    """Separate sensor corruption from case-mix effects and report descriptive uncertainty."""

    lines = [
        "# Frozen Model Robustness Experiment",
        "",
        f"- Source training run: {result['source_run_id']}",
        f"- Source split: {result['split_strategy']}",
        f"- Selected model: {result['selected_model']}",
        f"- Original test rows: {result['test_rows']}",
        f"- Seed: {result['seed']}; stochastic repetitions: {result['repeats']}",
        "- Model selection, parameters, and labels stay frozen.",
        "",
        "| Scenario | Repeats | Weighted F1 mean +/- std | F1 delta | Risk recall | "
        "Changed predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, scenario in result["scenarios"].items():
        if scenario["status"] != "completed":
            lines.append(f"| {name} (unavailable) | 0 | N/A | N/A | N/A | N/A |")
            continue
        f1 = scenario["metrics"]["f1_weighted"]
        lines.append(
            f"| {name} | {scenario['repetitions']} | {_format_score(f1['mean'])} "
            f"+/- {_format_score(f1['std'])} | "
            f"{_format_score(scenario['delta_from_baseline']['f1_weighted']['mean'])} | "
            f"{_format_score(scenario['metrics']['recall_risk']['mean'])} | "
            f"{_format_score(scenario['prediction_change_rate']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Sample Composition",
            "",
            "| Scenario | Missing feature fraction | Risk fraction | High-pollution fraction |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, scenario in result["scenarios"].items():
        if scenario["status"] == "completed":
            lines.append(
                f"| {name} | {_format_score(scenario['missing_fraction']['mean'])} | "
                f"{_format_score(scenario['risk_fraction']['mean'])} | "
                f"{_format_score(scenario['high_pollution_fraction']['mean'])} |"
            )
        else:
            lines.extend(["", f"Unavailable {name}: {scenario['reason']}"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Negative deltas mean lower scores; stresses can also improve scores by chance.",
            "Missingness/noise/bias perturb measurements while holding underlying "
            "case labels fixed.",
            "The high-pollution mix resamples unchanged held-out cases and labels together; "
            "its delta "
            "reflects case composition and sampling variation, not changed sensor accuracy.",
            "Standard deviations describe perturbation/resampling variability for one frozen "
            "model and holdout. They are not confidence intervals, retraining variability, "
            "or significance tests.",
            "Absent risk cases produce unavailable recall; observed repetition counts are "
            "saved in JSON.",
            "Scenarios and severity levels are fixed before scoring; do not tune a model "
            "on these results "
            "and reuse this holdout as independent evidence.",
            "Synthetic labels do not establish observed-failure performance. "
            "Real representative data "
            "with sourced labels remains necessary.",
            "",
        ]
    )
    return "\n".join(lines)


def run_robustness(
    training_run: str | Path,
    *,
    output_dir: str | Path = RUNS_DIR,
    seed: int = RANDOM_STATE,
    repeats: int = 5,
) -> dict:
    """Evaluate a completed run without fitting a model or changing its original artifacts."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer.")
    reference = load_evaluation_reference(training_run)
    config = {
        **reference.identity,
        "seed": seed,
        "repeats": repeats,
        "scenarios": [scenario.settings() for scenario in SCENARIOS],
    }
    with artifact_run(output_dir, "robustness", config) as artifacts:
        result = _run_scenarios(reference, artifacts, seed, repeats)
        plot_robustness_scores(result["scenarios"], artifacts.figures / "robustness_f1.png")
        save_json(result, artifacts.outputs / "robustness_summary.json")
        write_text_file(
            format_robustness_summary(result), artifacts.outputs / "robustness_summary.md"
        )
        return result
