"""Evaluation protocol: table checks, pairing, models, leave-one-drummer-out, summaries."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.conditions import BY_NAME
from src.features import MODEL_INPUTS, OPTIONAL_FEATURES
from src.models import evaluation
from src.models.classifiers import MODELS, MODELS_BY_NAME, ThresholdRule

CONDITION_NAMES = ("quantize_16_50", "quantize_32_100", "humanize_pink_matched")
CONDITIONS = tuple(BY_NAME[name] for name in CONDITION_NAMES)


@pytest.fixture
def table():
    """Three drummers x four clips x (human + three conditions); edits shrink timing."""
    rng = np.random.default_rng(0)
    rows = []
    for drummer in range(3):
        for clip in range(4):
            source = f"d{drummer}/take_clip{clip}"
            base = {name: rng.uniform(0.01, 0.03) for name in MODEL_INPUTS}
            for condition in ("human", *CONDITION_NAMES):
                factor = 1.0 if condition == "human" else 0.3
                rows.append({
                    "clip_id": f"{source}__{condition}", "source_clip_id": source,
                    "source_file": f"d{drummer}/take.mid", "drummer": f"d{drummer}",
                    "condition": condition,
                    **{name: value * factor for name, value in base.items()},
                    "tempo": 100.0 + drummer, "note_count": 30 + clip,
                    "style": "never a model input",
                })
    return pd.DataFrame(rows)


def validate(frame):
    evaluation.validate_table(frame, ("human", *CONDITION_NAMES))


# --- Table validation ------------------------------------------------------

def test_valid_table_passes(table):
    validate(table)


@pytest.mark.parametrize("column", ["clip_id", "source_clip_id", "drummer", "tempo", "mag_timing_std"])
def test_missing_column_is_rejected(table, column):
    with pytest.raises(ValueError, match="Missing"):
        validate(table.drop(columns=column))


def test_infinite_or_missing_required_feature_is_rejected(table):
    broken = table.copy()
    broken.loc[0, "mag_timing_std"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate(broken)
    broken.loc[0, "mag_timing_std"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        validate(broken)


def test_optional_features_may_be_missing(table):
    table.loc[0, list(OPTIONAL_FEATURES)] = np.nan
    validate(table)


@pytest.mark.parametrize("corruption", ["duplicate", "missing version", "drummer", "tempo", "one drummer"])
def test_broken_matching_is_rejected(table, corruption):
    if corruption == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    elif corruption == "missing version":
        table = table.iloc[1:]
    elif corruption == "one drummer":
        table = table[table.drummer == "d0"]
    else:
        table.loc[1, corruption] = "other" if corruption == "drummer" else 999.0
    with pytest.raises(ValueError):
        validate(table)


def test_pair_is_balanced(table):
    pair = evaluation.make_pair(table, "quantize_16_50")
    assert set(pair.condition) == {"human", "quantize_16_50"}
    assert pair.groupby("source_clip_id").size().eq(2).all()
    assert pair.groupby("condition").target.unique().map(list).to_dict() == {"human": [0], "quantize_16_50": [1]}
    with pytest.raises(ValueError):
        evaluation.make_pair(table, "human")


# --- Timing rule -------------------------------------------------------------

def test_timing_rule_learns_below_and_above():
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    below = ThresholdRule().fit(values, np.array([1, 1, 0, 0]))
    assert below.direction_ == "below" and below.predict(values).tolist() == [1, 1, 0, 0]
    above = ThresholdRule().fit(values, np.array([0, 0, 1, 1]))
    assert above.direction_ == "above" and above.predict(values).tolist() == [0, 0, 1, 1]


def test_timing_rule_ties_choose_below_and_the_lowest_cutoff():
    rule = ThresholdRule().fit(np.ones((4, 1)), np.array([0, 1, 0, 1]))
    assert rule.direction_ == "below" and rule.cutoff_ < 1.0
    assert rule.predict(np.ones((2, 1))).tolist() == [0, 0]


def test_timing_rule_separates_adjacent_floating_point_values():
    lower = np.nextafter(1.0, np.inf)
    values = np.array([[lower], [np.nextafter(lower, np.inf)]])
    assert ThresholdRule().fit(values, np.array([1, 0])).predict(values).tolist() == [1, 0]


# --- Leave-one-drummer-out ---------------------------------------------------

def test_every_model_is_evaluated_on_every_held_out_drummer(table):
    result = evaluation.evaluate(table, CONDITIONS, MODELS, verbose=False)
    folds, predictions = result.folds, result.predictions
    assert len(folds) == len(CONDITIONS) * len(MODELS) * 3
    assert folds.n_train.eq(16).all() and folds.n_test.eq(8).all()
    assert predictions.groupby(["condition", "model", "clip_id"]).size().eq(1).all()
    assert folds.groupby(["condition", "model"]).test_drummer.nunique().eq(3).all()
    assert folds[["tn", "fp", "fn", "tp"]].sum(axis=1).eq(folds.n_test).all()
    rule = folds[folds.model == "timing_rule"]
    assert rule.loc[rule.condition == "quantize_32_100", "rule_feature"].eq("mag_timing32_abs_mean").all()
    assert rule.loc[rule.condition != "quantize_32_100", "rule_feature"].eq("mag_timing_abs_mean").all()
    assert set(result.coefficients.model) == {"logistic_regression"}


def test_scaling_is_fitted_on_the_training_fold_only(table, monkeypatch):
    seen = []
    original_fit = StandardScaler.fit

    def record(scaler, values, *args, **kwargs):
        seen.append(tuple(np.asarray(values).mean(axis=0).round(12)))
        return original_fit(scaler, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record)
    model = MODELS_BY_NAME["logistic_regression"]
    evaluation.evaluate(table, CONDITIONS, (model,), verbose=False)
    expected = []
    for condition in CONDITIONS:
        pair = evaluation.make_pair(table, condition.name)
        for drummer in sorted(pair.drummer.unique()):
            train = pair.loc[pair.drummer != drummer, list(MODEL_INPUTS)]
            expected.append(tuple(train.to_numpy().mean(axis=0).round(12)))
    assert sorted(seen) == sorted(expected)


def test_feature_columns_can_be_restricted(table):
    columns = ("tempo", "mag_timing_std")
    model = MODELS_BY_NAME["logistic_regression"]
    result = evaluation.evaluate(table, CONDITIONS[:1], (model,), feature_columns=columns, verbose=False)
    assert set(result.coefficients.feature) == set(columns)


# --- Summaries -----------------------------------------------------------------

def test_summary_and_reliability(table):
    result = evaluation.evaluate(table, CONDITIONS, (MODELS_BY_NAME["timing_rule"],), verbose=False)
    summary = evaluation.summarize(result)
    assert len(summary) == len(CONDITIONS)
    assert summary.n_test.eq(24).all() and summary.n_drummers.eq(3).all()
    assert summary.reliable.equals(summary.drummer_min_f1 >= 0.9)


def test_reliability_thresholds():
    rows = []
    for name, reliable in [("quantize_16_25", True), ("quantize_16_50", False), ("quantize_16_75", True),
                           ("quantize_16_100", True), ("humanize_pink_5ms", True),
                           ("humanize_pink_10ms", True), ("humanize_pink_20ms", False),
                           ("humanize_pink_matched", True)]:
        rows.append({"condition": name, "model": "m", "reliable": reliable})
    thresholds = evaluation.reliability_thresholds(pd.DataFrame(rows)).set_index("series")
    # 25% is reliable but 50% is not, so quantization is reliable only from 75% on.
    assert thresholds.loc["quantize_16", "level"] == 0.75
    assert thresholds.loc["humanize_pink", "level"] == 10.0
