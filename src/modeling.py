"""Build supervised estimators with preprocessing fitted inside training folds."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import RANDOM_STATE
from src.feature_engineering import add_engineered_features


def build_classifier_pipeline(classifier, *, scale_features: bool = False) -> Pipeline:
    """Keep raw-value imputation, feature engineering, and prediction together."""

    # Preserve empty columns and names so every fold uses the same feature schema.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).set_output(
        transform="pandas"
    )
    # Compute flags and interactions after filling raw inputs with training medians.
    feature_engineering = FunctionTransformer(add_engineered_features)
    scaler = StandardScaler().set_output(transform="pandas") if scale_features else "passthrough"
    return Pipeline(
        [
            ("imputer", imputer),
            ("feature_engineering", feature_engineering),
            ("scaler", scaler),
            ("classifier", classifier),
        ]
    )


def build_model_candidates() -> dict[str, Pipeline]:
    """Scale logistic regression inputs; tree splits do not require scaling."""

    return {
        "random_forest": build_classifier_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )
        ),
        "logistic_regression": build_classifier_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            scale_features=True,
        ),
    }
