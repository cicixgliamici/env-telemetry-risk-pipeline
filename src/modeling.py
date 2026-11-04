"""Build supervised estimators with preprocessing fitted inside training folds."""

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import RANDOM_STATE
from src.decision import RiskDecisionPipeline
from src.feature_engineering import add_engineered_features


def build_classifier_pipeline(
    classifier, *, scale_features: bool = False, engineer_features: bool = True
) -> Pipeline:
    """Keep raw-value imputation, feature engineering, and prediction together."""

    # Preserve empty columns and names so every fold uses the same feature schema.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).set_output(
        transform="pandas"
    )
    # Compute flags and interactions after filling raw inputs with training medians.
    feature_engineering = (
        FunctionTransformer(add_engineered_features) if engineer_features else "passthrough"
    )
    scaler = StandardScaler().set_output(transform="pandas") if scale_features else "passthrough"
    return RiskDecisionPipeline(
        [
            ("imputer", imputer),
            ("feature_engineering", feature_engineering),
            ("scaler", scaler),
            ("classifier", classifier),
        ]
    )


def build_model_candidates() -> dict[str, Pipeline]:
    """Scale logistic regression inputs; tree splits do not require scaling."""

    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }
    candidates = {}
    # Pair each estimator with identical settings and folds to isolate feature engineering.
    for engineer_features in [True, False]:
        for name, classifier in classifiers.items():
            candidate_name = name if engineer_features else f"{name}_raw"
            candidates[candidate_name] = build_classifier_pipeline(
                clone(classifier),
                scale_features=name == "logistic_regression",
                engineer_features=engineer_features,
            )
    # A majority-class reference exposes models that do not improve on ignoring telemetry.
    candidates["dummy_most_frequent"] = build_classifier_pipeline(
        DummyClassifier(strategy="most_frequent"), engineer_features=False
    )
    return candidates
