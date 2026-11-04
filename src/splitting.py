"""Build auditable random or chronological evaluation partitions."""

from math import ceil, isfinite

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from src.config import RANDOM_STATE, TARGET_COLUMN

SPLIT_STRATEGIES = ("stratified_random", "temporal")


def validate_split_settings(strategy: str, test_size: float, temporal_gap: int) -> None:
    """Reject ambiguous settings before creating any evaluation partitions."""

    if strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"Unknown split strategy: {strategy}")
    if not isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("test_size must be a finite fraction strictly between 0 and 1.")
    if not isinstance(temporal_gap, int) or temporal_gap < 0:
        raise ValueError("temporal_gap must be a non-negative integer.")
    if strategy != "temporal" and temporal_gap:
        raise ValueError("temporal_gap requires the temporal split strategy.")


def utc_timestamps(timestamps: pd.Series) -> pd.Series:
    """Compare instants consistently; naive timestamps are interpreted as UTC."""

    parsed = pd.to_datetime(timestamps, utc=True, format="mixed", errors="raise")
    if parsed.isna().any():
        raise ValueError("Temporal evaluation requires complete timestamps.")
    return parsed


def temporal_holdout(
    dataset: pd.DataFrame, test_size: float, temporal_gap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest timestamp groups and exclude a gap before them."""

    ordered = dataset.copy()
    ordered["timestamp"] = utc_timestamps(ordered["timestamp"])
    ordered = ordered.sort_values("timestamp", kind="stable")
    times = ordered["timestamp"].drop_duplicates().to_numpy()
    test_start = len(times) - ceil(len(times) * test_size)
    train_stop = test_start - temporal_gap
    if train_stop < 1:
        raise ValueError("Temporal holdout leaves no training timestamps; reduce test_size or gap.")
    # Boundaries use whole instants, so simultaneous observations never straddle partitions.
    train = ordered[ordered["timestamp"] <= times[train_stop - 1]]
    test = ordered[ordered["timestamp"] >= times[test_start]]
    return train, test


def build_training_folds(
    target: pd.Series,
    *,
    strategy: str = "stratified_random",
    timestamps: pd.Series | None = None,
    temporal_gap: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize folds once so every candidate sees identical training observations."""

    if strategy == "stratified_random":
        folds = min(3, int(target.value_counts().min()))
        if folds < 2:
            raise ValueError(
                "Model selection requires at least two training rows per class "
                "for stratified cross-validation. Provide more labeled observations."
            )
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        return list(splitter.split(np.zeros(len(target)), target))
    if strategy != "temporal":
        raise ValueError(f"Unknown split strategy: {strategy}")
    return _temporal_folds(target, timestamps, temporal_gap)


def _temporal_folds(target, timestamps, temporal_gap):
    """Expand historical training windows without shuffling or splitting timestamp groups."""

    if timestamps is None or not timestamps.index.equals(target.index):
        raise ValueError("Temporal CV requires timestamps aligned with the training target.")
    times = utc_timestamps(timestamps)
    if not times.is_monotonic_increasing:
        raise ValueError("Temporal CV requires training observations in chronological order.")
    codes, unique_times = pd.factorize(times, sort=True)
    folds = min(3, len(unique_times) - 1)
    if folds < 2:
        raise ValueError("Temporal CV requires at least three distinct training timestamps.")
    splitter = TimeSeriesSplit(n_splits=folds, gap=temporal_gap)
    try:
        group_splits = list(splitter.split(unique_times))
    except ValueError as exc:
        raise ValueError("Temporal CV has insufficient history for the requested gap.") from exc
    splits = []
    for number, (train_groups, validation_groups) in enumerate(group_splits, start=1):
        train = np.flatnonzero(np.isin(codes, train_groups))
        validation = np.flatnonzero(np.isin(codes, validation_groups))
        # All candidates must be trainable; do not silently skip folds based on their scores.
        if target.iloc[train].nunique() < 2:
            raise ValueError(
                f"Temporal CV fold {number} requires at least two classes in its training "
                "history. Provide more representative historical data."
            )
        splits.append((train, validation))
    return splits


def _partition_record(dataset: pd.DataFrame, indices) -> dict:
    """Identify rows by position in the cleaned export, retaining their time range."""

    partition = dataset.loc[indices]
    times = utc_timestamps(partition["timestamp"])
    return {
        "row_indices": [int(index) for index in indices],
        "rows": len(partition),
        "timestamp_min": times.min().isoformat(),
        "timestamp_max": times.max().isoformat(),
        "class_counts": {
            str(label): int(count)
            for label, count in partition[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }


def build_evaluation_protocol(dataset, train, test, folds, strategy, test_size, temporal_gap):
    """Describe the exact holdout and CV membership without changing their selection."""

    return {
        "split_strategy": strategy,
        "cv_strategy": "expanding_window" if strategy == "temporal" else "stratified_kfold",
        "requested_test_size": test_size,
        "test_size_unit": "distinct_timestamps" if strategy == "temporal" else "rows",
        "temporal_gap": temporal_gap,
        "gap_unit": "distinct_timestamps",
        "row_index_reference": "processed/cleaned_dataset.csv (zero-based data row)",
        "training": _partition_record(dataset, train.index),
        "test": _partition_record(dataset, test.index),
        "excluded_row_indices": [
            int(index)
            for index in dataset.index
            if index not in train.index and index not in test.index
        ],
        "folds": [
            {
                "fold": number,
                "training": _partition_record(dataset, train.index[train_positions]),
                "validation": _partition_record(dataset, train.index[validation_positions]),
            }
            for number, (train_positions, validation_positions) in enumerate(folds, start=1)
        ],
    }
