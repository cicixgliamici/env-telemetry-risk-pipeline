"""Generate fixed measurement stresses and a changed case mix using training-only scales."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.config import NUMERIC_RANGES


@dataclass(frozen=True)
class StressScenario:
    """Keep scenario definitions serializable and independent of observed test scores."""

    name: str
    kind: str
    severity: float
    random_stream: int = 0

    def settings(self) -> dict:
        return asdict(self)


SCENARIOS = (
    StressScenario("baseline", "baseline", 0),
    StressScenario("missing_10pct", "missing", 0.1, 10),
    StressScenario("missing_30pct", "missing", 0.3, 10),
    StressScenario("noise_0.1std", "noise", 0.1, 20),
    StressScenario("noise_0.3std", "noise", 0.3, 20),
    StressScenario("sensor_bias_0.5std", "bias", 0.5),
    StressScenario("high_pollution_mix_3x", "case_mix", 3, 30),
)


def training_stress_profile(features: pd.DataFrame) -> dict:
    """Derive every perturbation scale and sampling threshold from historical training rows."""

    # A constant or entirely missing training feature has zero noise/bias scale.
    scales = features.std(ddof=0).fillna(0)
    if not np.isfinite(scales).all():
        raise ValueError("Training telemetry scales must be finite for robustness experiments.")
    return {
        "noise_std": {column: float(value) for column, value in scales.items()},
        "pollution_median": float(features["pollution_proxy"].median())
        if features["pollution_proxy"].notna().any()
        else None,
        "pollution_upper_quartile": float(features["pollution_proxy"].quantile(0.75))
        if features["pollution_proxy"].notna().any()
        else None,
        "bias_columns": ["humidity_pct", "temperature_c", "pollution_proxy"],
        "scale_source": "holdout_training_rows_only",
    }


def _clip_physical_ranges(features: pd.DataFrame) -> pd.DataFrame:
    """Keep synthetic sensor errors within the same broad bounds as accepted telemetry."""

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        features[column] = features[column].clip(lower=minimum, upper=maximum)
    features["pressure_hpa"] = features["pressure_hpa"].clip(lower=np.nextafter(0.0, 1.0))
    return features


def scenario_cases(
    features: pd.DataFrame, profile: dict, scenario: StressScenario, seed: int, repetition: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return changed measurements and their original held-out row positions."""

    changed = features.copy()
    positions = np.arange(len(features))
    # Shared streams nest missing-value masks and scale the same noise across severities.
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario.random_stream, repetition]))
    if scenario.kind == "missing":
        changed = changed.mask(rng.random(features.shape) < scenario.severity)
    elif scenario.kind == "noise":
        scales = np.array([profile["noise_std"][column] for column in features.columns])
        changed = changed + rng.standard_normal(features.shape) * scales * scenario.severity
        changed = _clip_physical_ranges(changed)
    elif scenario.kind == "bias":
        for column in profile["bias_columns"]:
            changed[column] = changed[column] + profile["noise_std"][column] * scenario.severity
        changed = _clip_physical_ranges(changed)
    elif scenario.kind == "case_mix":
        if profile["pollution_upper_quartile"] is None:
            raise ValueError("Case-mix experiments require observed training pollution values.")
        pollution = features["pollution_proxy"].fillna(profile["pollution_median"])
        high = pollution > profile["pollution_upper_quartile"]
        weights = np.where(high, scenario.severity, 1.0)
        positions = rng.choice(
            positions, size=len(features), replace=True, p=weights / weights.sum()
        )
        # Resample features and labels together; this changes case mix, not sensor readings.
        changed = features.iloc[positions].copy()
    elif scenario.kind != "baseline":
        raise ValueError(f"Unknown stress scenario kind: {scenario.kind}")
    return changed.reset_index(drop=True), positions
