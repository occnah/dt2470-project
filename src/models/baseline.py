"""Fixed logistic and timing-threshold baselines with held-out drummers."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import HUMAN, QUANTIZATION_CONDITIONS, SEED


FEATURE_COLUMNS = (
    "note_count", "timing_abs_mean", "timing_std", "ioi_mean", "ioi_std",
    "velocity_mean", "velocity_std", "velocity_range",
    "timing_32_abs_mean", "timing_32_std",
)
EXPERIMENT_CONDITIONS = tuple(c[0] for c in QUANTIZATION_CONDITIONS) + ("velocity_normalized",)
MODEL_NAMES = ("logistic_regression", "timing_threshold")
METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1")


def validate_table(frame: pd.DataFrame) -> None:
    """Reject incomplete pairs, ambiguous grouping, and unusable feature values."""
    identifiers = ("clip_id", "source_clip_id", "source_file", "drummer", "condition")
    required = set(identifiers + FEATURE_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The feature table is empty")
    for name in identifiers:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or empty {name}")
    if not frame.clip_id.is_unique:
        raise ValueError("Duplicate clip_id values")
    if frame.duplicated(["source_clip_id", "condition"]).any():
        raise ValueError("Duplicate source_clip_id/condition pairs")
    expected = {HUMAN, *EXPERIMENT_CONDITIONS}
    if set(frame.condition) != expected:
        raise ValueError(f"Expected exactly these seven conditions: {sorted(expected)}")
    grouped = frame.groupby("source_clip_id")
    if not grouped.size().eq(len(expected)).all():
        raise ValueError("Every source_clip_id must have all seven matched conditions")
    for name in ("drummer", "source_file", "note_count"):
        if grouped[name].nunique().gt(1).any():
            raise ValueError(f"Inconsistent {name} across versions of a source clip")
    if frame.groupby("source_file").drummer.nunique().gt(1).any():
        raise ValueError("A source_file cannot belong to multiple drummers")
    if frame.drummer.nunique() < 2:
        raise ValueError("At least two drummers are required")
    try:
        values = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Features must be numeric and finite") from error
    if not np.isfinite(values).all():
        raise ValueError("Features contain missing or nonfinite values")


def make_pair(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Return balanced matched human/condition rows; transformed is target 1."""
    if condition not in EXPERIMENT_CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    pair = frame[frame.condition.isin([HUMAN, condition])].copy()
    if (
        pair.empty
        or pair.duplicated(["source_clip_id", "condition"]).any()
        or not pair.groupby("source_clip_id").size().eq(2).all()
    ):
        raise ValueError("Each source clip needs exactly one human and one transformed row")
    pair["target"] = pair.condition.eq(condition).astype(int)
    return pair.reset_index(drop=True)


def make_classifier():
    """Use a fixed model; its scaler is fitted inside each training fold."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED),
    )


def fit_timing_threshold(values: np.ndarray, targets: np.ndarray) -> float:
    """Maximize training balanced accuracy for transformed iff value <= cutoff.

    Candidates separate distinct training values and include both constant
    predictions. Equal objectives (within 1e-12) choose the lowest cutoff.
    """
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets)
    if values.ndim != 1 or targets.shape != values.shape or not len(values):
        raise ValueError("Threshold fitting requires equally sized nonempty vectors")
    if not np.isfinite(values).all() or set(np.unique(targets)) != {0, 1}:
        raise ValueError("Threshold fitting requires finite values and both binary classes")
    order = np.argsort(values, kind="stable")
    ordered_values, ordered_targets = values[order], targets[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_values)), len(values) - 1]
    unique = ordered_values[ends]
    midpoints = unique[:-1] / 2 + unique[1:] / 2
    midpoints = np.where(midpoints >= unique[1:], unique[:-1], midpoints)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        midpoints,
        unique[-1],
    ]
    tp = np.r_[0, np.cumsum(ordered_targets)[ends]]
    fp = np.r_[0, np.cumsum(1 - ordered_targets)[ends]]
    positives, negatives = np.sum(targets), np.sum(1 - targets)
    objective = 0.5 * (tp / positives + (negatives - fp) / negatives)
    best = np.flatnonzero(np.isclose(objective, objective.max(), rtol=0, atol=1e-12))[0]
    return float(candidates[best])


def binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Positive-class metrics and a human/transformed confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="binary", pos_label=1, zero_division=0,
    )
    return {
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_table(
    frame: pd.DataFrame,
    conditions: tuple[str, ...] = EXPERIMENT_CONDITIONS,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the given experiments, testing every row once per method/experiment.

    `conditions` and `feature_columns` default to the full fixed baseline
    protocol (all six experiments, all ten features). Either can be
    narrowed for an ablation — e.g. the four 1/16 experiments with the
    1/32 timing features removed, to check whether their relationship to
    the 1/16 deviation at 50%+ quantization strength (see WHATS_LEFT.md)
    is giving logistic regression a shortcut — without duplicating this
    function's leakage checks or threshold fitting. `feature_columns`
    only changes what logistic regression is trained/predicted on; the
    timing-threshold baseline's single `threshold_feature` is unaffected
    unless it's also been excluded.
    """
    validate_table(frame)
    folds, predictions = [], []
    for condition in conditions:
        pair = make_pair(frame, condition)
        threshold_feature = "timing_32_abs_mean" if condition == "quantized_32_100" else "timing_abs_mean"
        splitter = LeaveOneGroupOut()
        for train_index, test_index in splitter.split(pair, pair.target, groups=pair.drummer):
            train, test = pair.iloc[train_index], pair.iloc[test_index]
            for name in ("drummer", "source_file", "source_clip_id"):
                if set(train[name]) & set(test[name]):
                    raise ValueError(f"Train/test leakage in {name}")
            train_y, test_y = train.target.to_numpy(), test.target.to_numpy()
            classifier = make_classifier()
            classifier.fit(train[list(feature_columns)], train_y)
            logistic_predictions = classifier.predict(test[list(feature_columns)])
            threshold = fit_timing_threshold(train[threshold_feature].to_numpy(), train_y)
            threshold_predictions = (test[threshold_feature].to_numpy() <= threshold).astype(int)
            for model, predicted, cutoff, feature in (
                ("logistic_regression", logistic_predictions, 0.5, "predicted_probability"),
                ("timing_threshold", threshold_predictions, threshold, threshold_feature),
            ):
                folds.append({
                    "condition": condition, "model": model,
                    "test_drummer": str(test.drummer.iloc[0]),
                    "train_drummers": "|".join(sorted(train.drummer.unique())),
                    "n_train": len(train), "n_test": len(test),
                    "threshold": cutoff, "threshold_feature": feature,
                    **binary_metrics(test_y, predicted),
                })
                rows = test[["drummer", "clip_id", "source_clip_id"]].copy()
                rows.insert(0, "condition", condition)
                rows.insert(1, "model", model)
                rows["y_true"], rows["y_pred"] = test_y, predicted
                predictions.append(rows)
        print(f"Completed {condition}: {pair.drummer.nunique()} held-out drummers", flush=True)
    return pd.DataFrame(folds), pd.concat(predictions, ignore_index=True)


def summarize_metrics(
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    conditions: tuple[str, ...] = EXPERIMENT_CONDITIONS,
) -> pd.DataFrame:
    """Report pooled predictions and equal-weight drummer means/sample SDs.

    `conditions` must match whatever subset `evaluate_table` was run with
    (defaults to all six); summarizing a condition with no rows in `folds`/
    `predictions` would otherwise hit an empty-input error below.
    """
    summary = []
    for condition in conditions:
        for model in MODEL_NAMES:
            test = predictions[(predictions.condition == condition) & (predictions.model == model)]
            subset = folds[(folds.condition == condition) & (folds.model == model)]
            row = {
                "condition": condition, "model": model, "n_test": len(test),
                "n_drummers": len(subset), **binary_metrics(test.y_true, test.y_pred),
            }
            for metric in METRIC_COLUMNS:
                row[f"drummer_mean_{metric}"] = float(subset[metric].mean())
                row[f"drummer_sd_{metric}"] = float(subset[metric].std(ddof=1))
            row["drummer_min_f1"] = float(subset.f1.min())
            row["drummer_max_f1"] = float(subset.f1.max())
            summary.append(row)
    return pd.DataFrame(summary)
