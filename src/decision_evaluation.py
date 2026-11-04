"""Persist training-only decision evidence and compare two frozen rules on the holdout."""

from src.decision import binary_error_cost, select_cost_threshold, threshold_predictions
from src.evaluate import evaluate_predictions
from src.utils import save_json
from src.visualize import plot_threshold_selection


def configure_decision_policy(model, features, target, cv_splits, config, artifacts):
    """Choose and attach a policy before fitting on all training rows or inspecting the test set."""

    if config["decision_policy"] == "classifier":
        # Reused estimators must not retain a cutoff selected by an earlier training run.
        model.__dict__.pop("risk_decision_policy_", None)
        return {"mode": "classifier"}
    policy, curve, predictions = select_cost_threshold(
        model,
        features,
        target,
        cv_splits,
        config["false_negative_cost"],
        config["false_positive_cost"],
    )
    # Learned state travels with the artifact; no scoring-time metadata sidecar is needed.
    model.risk_decision_policy_ = policy
    curve.to_csv(artifacts.outputs / "threshold_comparison.csv", index=False)
    predictions.to_csv(artifacts.outputs / "threshold_validation_predictions.csv", index=False)
    save_json(policy, artifacts.outputs / "decision_policy.json")
    plot_threshold_selection(curve, policy, artifacts.figures / "threshold_selection.png")
    return policy


def evaluate_model_decision(model, test_features, test_target, policy):
    """Score preselected rules once; test results cannot revise the selected threshold."""

    # Reuse one probability matrix for both predeclared decisions on exactly the same cases.
    probabilities = model.predict_proba(test_features)
    classifier_predictions = model.classes_[probabilities.argmax(axis=1)]
    selected_predictions = threshold_predictions(probabilities, model.classes_, policy["threshold"])
    comparisons = {}
    for name, labels in [
        ("classifier", classifier_predictions),
        ("selected_policy", selected_predictions),
    ]:
        comparisons[name] = {
            **binary_error_cost(
                test_target,
                labels,
                policy["false_negative_cost"],
                policy["false_positive_cost"],
                policy["other_label"],
            ),
            "metrics": evaluate_predictions(test_target, labels),
        }
    return (
        selected_predictions,
        dict(comparisons["selected_policy"]["metrics"]),
        {"decision_evaluation": comparisons},
    )


def decision_summary_lines(metrics: dict) -> list[str]:
    policy = metrics["decision_policy"]
    if policy["mode"] == "classifier":
        return ["- Decision policy: classifier's original decision rule"]
    threshold = (
        "never risk" if policy["threshold"] is None else f"p(risk) >= {policy['threshold']:.2f}"
    )
    lines = [
        "## Cost-Based Decision Policy",
        "",
        f"- Selected rule: {threshold}",
        f"- Relative error costs: FN={policy['false_negative_cost']}, "
        f"FP={policy['false_positive_cost']}",
        "- Threshold selection: pooled training-only validation probabilities; "
        "no test-score tuning.",
        f"- Validation coverage: {policy['validation_rows']} of "
        f"{policy['training_rows']} training rows",
        "- Candidate CV scores use original classifier decisions; "
        "headline test scores use the selected rule.",
        "- Costs are user-supplied relative weights; probability calibration is not assumed.",
        "",
        "| Held-out rule | FN | FP | Risk precision | Risk recall | Cost per observation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in metrics["decision_evaluation"].items():
        precision = "N/A" if values["precision_risk"] is None else f"{values['precision_risk']:.4f}"
        recall = "N/A" if values["recall_risk"] is None else f"{values['recall_risk']:.4f}"
        lines.append(
            f"| {name} | {values['false_negatives']} | {values['false_positives']} | "
            f"{precision} | {recall} | {values['cost_per_observation']:.4f} |"
        )
    return lines
