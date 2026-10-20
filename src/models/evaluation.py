"""
The evaluation protocol shared by all experiments.

Each experiment is one balanced binary task: the human version of every
clip (class 0) against one edited condition of the same clips (class 1).
Models are evaluated leave-one-drummer-out: train on all drummers but one,
test on the held-out drummer, for every drummer in turn. All versions of a
clip and all clips of a performance belong to one drummer, so nothing seen
in training appears in the test fold in any form.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut

from src.common.config import HUMAN, RELIABLE_F1
from src.data.conditions import BY_NAME, Condition
from src.features import MODEL_INPUTS, OPTIONAL_FEATURES
from src.models.classifiers import MODELS, ModelSpec

ID_COLUMNS = ("clip_id", "source_clip_id", "source_file", "drummer", "condition")
# Properties of the source clip that every version must share.
SHARED_COLUMNS = ("drummer", "source_file", "note_count", "tempo")
METRICS = ("accuracy", "precision", "recall", "f1")


def validate_table(
    frame: pd.DataFrame,
    conditions: tuple[str, ...],
    feature_columns: tuple[str, ...] = MODEL_INPUTS,
) -> None:
    """
    Reject a table that would make the evaluation invalid: missing columns
    or IDs, duplicate rows, source clips without every requested condition,
    versions of one clip that disagree about their source, or unusable
    feature values. Raises ValueError.
    """
    missing = set(ID_COLUMNS + SHARED_COLUMNS + tuple(feature_columns)) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The feature table is empty")
    for name in ID_COLUMNS:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or empty {name}")
    if not frame.clip_id.is_unique:
        raise ValueError("Duplicate clip_id values")

    rows = frame[frame.condition.isin(conditions)]
    if set(rows.condition) != set(conditions):
        raise ValueError(f"Conditions missing from the table: {sorted(set(conditions) - set(rows.condition))}")
    if rows.duplicated(["source_clip_id", "condition"]).any():
        raise ValueError("Duplicate source_clip_id/condition pairs")
    grouped = rows.groupby("source_clip_id")
    if not grouped.size().eq(len(conditions)).all():
        raise ValueError("Every source clip must have exactly one row per condition")
    for name in SHARED_COLUMNS:
        if grouped[name].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent {name} across versions of a source clip")
    if rows.groupby("source_file").drummer.nunique().gt(1).any():
        raise ValueError("A source_file cannot belong to multiple drummers")
    if rows.drummer.nunique() < 2:
        raise ValueError("At least two drummers are required")

    try:
        values = rows[list(feature_columns)].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Features must be numeric") from error
    if np.isinf(values).any():
        raise ValueError("Features contain infinite values")
    required = [c for c in feature_columns if c not in OPTIONAL_FEATURES]
    if rows[required].isna().any().any():
        raise ValueError("Missing values in features that must always be defined")


def make_pair(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Human rows (target 0) and `condition` rows (target 1) of the same clips."""
    if condition == HUMAN:
        raise ValueError("The edited condition cannot be human")
    pair = frame[frame.condition.isin([HUMAN, condition])].copy()
    if pair.empty or not pair.groupby("source_clip_id").size().eq(2).all():
        raise ValueError("Each source clip needs exactly one human and one edited row")
    pair["target"] = pair.condition.eq(condition).astype(int)
    return pair.reset_index(drop=True)


def binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    """Accuracy, and precision/recall/F1 for the edited class, plus confusion counts."""
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average="binary", pos_label=1, zero_division=0,
    )
    return {
        "accuracy": float(np.mean(np.asarray(targets) == np.asarray(predictions))),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@dataclass
class EvaluationResult:
    """
    folds:        one row per condition, model and held-out drummer
    predictions:  one row per condition, model and test clip
    coefficients: standardized logistic-regression coefficients per fold
    """

    folds: pd.DataFrame
    predictions: pd.DataFrame
    coefficients: pd.DataFrame


def evaluate(
    frame: pd.DataFrame,
    conditions: tuple[Condition, ...],
    models: tuple[ModelSpec, ...] = MODELS,
    feature_columns: tuple[str, ...] = MODEL_INPUTS,
    verbose: bool = True,
) -> EvaluationResult:
    """Run leave-one-drummer-out for every condition and model."""
    used_columns = sorted({c for m in models for cond in conditions for c in m.inputs(cond, feature_columns)})
    validate_table(frame, (HUMAN, *(c.name for c in conditions)), tuple(used_columns))

    folds, predictions, coefficients = [], [], []
    for condition in conditions:
        pair = make_pair(frame, condition.name)
        for train_index, test_index in LeaveOneGroupOut().split(pair, groups=pair.drummer):
            train, test = pair.iloc[train_index], pair.iloc[test_index]
            for name in ("drummer", "source_file", "source_clip_id"):
                if set(train[name]) & set(test[name]):
                    raise ValueError(f"Train/test leakage in {name}")
            test_drummer = str(test.drummer.iloc[0])

            for model in models:
                columns = list(model.inputs(condition, feature_columns))
                estimator = model.build()
                estimator.fit(train[columns], train.target)
                predicted = np.asarray(estimator.predict(test[columns]), dtype=int)

                fold = {
                    "condition": condition.name, "model": model.name,
                    "test_drummer": test_drummer, "n_train": len(train), "n_test": len(test),
                    **binary_metrics(test.target, predicted),
                }
                if hasattr(estimator, "cutoff_"):
                    fold["rule_feature"], fold["rule_cutoff"] = columns[0], estimator.cutoff_
                    fold["rule_direction"] = estimator.direction_
                folds.append(fold)

                rows = test[["drummer", "clip_id", "source_clip_id"]].copy()
                rows.insert(0, "condition", condition.name)
                rows.insert(1, "model", model.name)
                rows["y_true"], rows["y_pred"] = test.target.to_numpy(), predicted
                predictions.append(rows)

                final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
                if hasattr(final, "coef_"):
                    for feature, value in zip(columns, final.coef_[0]):
                        coefficients.append({
                            "condition": condition.name, "model": model.name,
                            "test_drummer": test_drummer, "feature": feature, "coefficient": value,
                        })
        if verbose:
            print(f"  {condition.name}: {pair.drummer.nunique()} held-out drummers", flush=True)

    return EvaluationResult(
        pd.DataFrame(folds),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(coefficients, columns=["condition", "model", "test_drummer", "feature", "coefficient"]),
    )


def summarize(result: EvaluationResult) -> pd.DataFrame:
    """
    One row per condition and model: pooled metrics over all held-out
    predictions, and the equal-weight mean, SD (ddof=1), minimum and
    maximum over held-out drummers. `reliable` is True when F1 is at least
    RELIABLE_F1 for every held-out drummer.
    """
    pooled = result.predictions.groupby(["condition", "model"], sort=False).apply(
        lambda rows: pd.Series(binary_metrics(rows.y_true, rows.y_pred)), include_groups=False,
    )
    per_drummer = result.folds.groupby(["condition", "model"], sort=False)
    spread = pd.concat(
        [
            per_drummer[list(METRICS)].mean().add_prefix("drummer_mean_"),
            per_drummer[list(METRICS)].std(ddof=1).add_prefix("drummer_sd_"),
            per_drummer.f1.min().rename("drummer_min_f1"),
            per_drummer.f1.max().rename("drummer_max_f1"),
            per_drummer.size().rename("n_drummers"),
        ],
        axis=1,
    )
    summary = pooled.join(spread).reset_index()
    summary["n_test"] = summary[["tn", "fp", "fn", "tp"]].sum(axis=1)
    summary["reliable"] = summary.drummer_min_f1 >= RELIABLE_F1
    return summary


def reliability_thresholds(summary: pd.DataFrame) -> pd.DataFrame:
    """
    For each model and condition series, the level at which detection
    becomes reliable and stays reliable:

    - quantize / velocity: the lowest strength from which every stronger
      level is reliable ("reliable from")
    - humanize: the largest jitter up to which every smaller jitter is
      reliable ("reliable up to"); the matched level is reported separately

    NaN means no level qualifies.
    """
    rows = []
    for model, model_rows in summary.groupby("model", sort=False):
        reliable = dict(zip(model_rows.condition, model_rows.reliable))
        series = {}
        for name in reliable:
            condition = BY_NAME[name]
            if condition.family in ("quantize", "velocity") and condition.strength is not None:
                if condition.family == "quantize" and condition.subdivision != 16:
                    continue
                series.setdefault(condition.series, []).append((condition.strength, reliable[name]))
            elif condition.family == "humanize" and not condition.matched:
                series.setdefault(condition.series, []).append((condition.jitter_sd_ms, reliable[name]))
        for name, levels in series.items():
            levels.sort()
            if name.startswith("humanize"):
                qualifying = [level for i, (level, _) in enumerate(levels) if all(ok for _, ok in levels[: i + 1])]
                rows.append({"model": model, "series": name, "rule": "reliable up to",
                             "level": qualifying[-1] if qualifying else np.nan})
            else:
                qualifying = [level for i, (level, _) in enumerate(levels) if all(ok for _, ok in levels[i:])]
                rows.append({"model": model, "series": name, "rule": "reliable from",
                             "level": qualifying[0] if qualifying else np.nan})
    return pd.DataFrame(rows, columns=["model", "series", "rule", "level"])
