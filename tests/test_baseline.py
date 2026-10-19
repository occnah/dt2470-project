import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import QUANTIZATION_CONDITIONS
from src.models import baseline


@pytest.fixture
def paired_table():
    rows = []
    conditions = ("human", *baseline.EXPERIMENT_CONDITIONS)
    strengths = {name: strength for name, _, strength in QUANTIZATION_CONDITIONS}
    for drummer in range(3):
        for clip in range(2):
            source_id = f"d{drummer}/take_clip{clip}"
            for condition in conditions:
                timing = 0.02 + drummer * 0.003 + clip * 0.001
                strength = strengths.get(condition, 0.0)
                normalized = condition == "velocity_normalized"
                rows.append({
                    "clip_id": f"{source_id}__{condition}",
                    "source_clip_id": source_id,
                    "source_file": f"d{drummer}/take.mid",
                    "drummer": f"d{drummer}",
                    "condition": condition,
                    "note_count": 10 + drummer * 20 + clip,
                    "timing_abs_mean": timing * (1.0 - strength),
                    "timing_std": timing * (1.0 - strength),
                    "timing_32_abs_mean": timing * (1.0 - strength) / 2,
                    "timing_32_std": timing * (1.0 - strength) / 2,
                    "ioi_mean": 0.1 + clip * 0.01,
                    "ioi_std": 0.03,
                    "velocity_mean": 80.0,
                    "velocity_std": 0.0 if normalized else 10.0,
                    "velocity_range": 0.0 if normalized else 40.0,
                    "irrelevant_metadata": "never feed this text to the model",
                })
    return pd.DataFrame(rows)


def test_only_measurements_are_classifier_features():
    assert set(baseline.FEATURE_COLUMNS) == {
        "note_count", "timing_abs_mean", "timing_std", "timing_32_abs_mean",
        "timing_32_std", "ioi_mean", "ioi_std", "velocity_mean",
        "velocity_std", "velocity_range",
    }
    assert baseline.EXPERIMENT_CONDITIONS == (
        *(name for name, _, _ in QUANTIZATION_CONDITIONS), "velocity_normalized",
    )
    pipeline = baseline.make_classifier()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert isinstance(pipeline.steps[0][1], StandardScaler)
    classifier = pipeline.steps[1][1]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.C == 1.0
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 2000
    assert classifier.random_state == 42


@pytest.mark.parametrize("column", [
    "clip_id", "source_clip_id", "source_file", "drummer", "condition",
    *baseline.FEATURE_COLUMNS,
])
def test_missing_required_column_is_rejected(paired_table, column):
    with pytest.raises(ValueError):
        baseline.validate_table(paired_table.drop(columns=column))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_feature_is_rejected(paired_table, value):
    paired_table.loc[0, "timing_abs_mean"] = value
    with pytest.raises(ValueError):
        baseline.validate_table(paired_table)


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "drummer", "source_file"])
def test_broken_matching_is_rejected(paired_table, corruption):
    if corruption == "duplicate":
        paired_table = pd.concat([paired_table, paired_table.iloc[[0]]], ignore_index=True)
    elif corruption == "missing":
        paired_table = paired_table.iloc[1:]
    else:
        paired_table.loc[1, corruption] = "wrong_source"
    with pytest.raises(ValueError):
        baseline.validate_table(paired_table)


def test_pair_is_balanced_and_preserves_matching(paired_table):
    baseline.validate_table(paired_table)
    pair = baseline.make_pair(paired_table, "quantized_16_25")
    assert set(pair.condition) == {"human", "quantized_16_25"}
    assert pair.groupby("source_clip_id").size().eq(2).all()
    assert pair.groupby("condition").target.unique().map(list).to_dict() == {
        "human": [0], "quantized_16_25": [1],
    }
    assert "target" not in paired_table.columns


def test_threshold_has_correct_direction_and_earliest_optimal_split():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    threshold = baseline.fit_timing_threshold(values, np.array([1, 0, 1, 0]))
    assert 1.0 <= threshold < 2.0
    assert baseline.fit_timing_threshold(values[::-1], np.array([0, 1, 0, 1])) == threshold
    perfect = baseline.fit_timing_threshold(values, np.array([1, 1, 0, 0]))
    np.testing.assert_array_equal(values <= perfect, [True, True, False, False])


def test_threshold_can_predict_all_negative_when_timing_has_no_information():
    threshold = baseline.fit_timing_threshold(np.ones(4), np.array([0, 1, 0, 1]))
    assert threshold < 1.0


def test_threshold_separates_adjacent_floating_point_values():
    lower = np.nextafter(1.0, np.inf)
    upper = np.nextafter(lower, np.inf)
    values = np.array([lower, upper])
    threshold = baseline.fit_timing_threshold(values, np.array([1, 0]))
    np.testing.assert_array_equal(values <= threshold, [True, False])


def test_evaluation_holds_out_each_drummer_and_fits_scaling_on_training_only(paired_table, monkeypatch):
    fitted_means = []
    original_fit = StandardScaler.fit

    def record_fit(scaler, values, *args, **kwargs):
        fitted_means.append(tuple(np.asarray(values).mean(axis=0).round(12)))
        return original_fit(scaler, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    metrics, predictions = baseline.evaluate_table(paired_table)
    assert len(metrics) == 3 * 6 * 2
    assert len(predictions) == 6 * 2 * 6 * 2
    assert set(metrics.model) == {"logistic_regression", "timing_threshold"}
    assert metrics.n_train.eq(8).all() and metrics.n_test.eq(4).all()
    assert predictions.groupby(["condition", "model", "clip_id"]).size().eq(1).all()
    assert metrics.groupby(["condition", "model"]).test_drummer.nunique().eq(3).all()
    assert metrics[["accuracy", "precision", "recall", "f1"]].ge(0).all().all()
    assert metrics[["accuracy", "precision", "recall", "f1"]].le(1).all().all()
    assert metrics[["tn", "fp", "fn", "tp"]].sum(axis=1).eq(metrics.n_test).all()
    expected_means = []
    for condition in baseline.EXPERIMENT_CONDITIONS:
        pair = baseline.make_pair(paired_table, condition)
        for drummer in paired_table.drummer.unique():
            train = pair.loc[pair.drummer != drummer, list(baseline.FEATURE_COLUMNS)]
            expected_means.append(tuple(train.to_numpy().mean(axis=0).round(12)))
    assert sorted(fitted_means) == sorted(expected_means)
    for row in metrics.itertuples():
        assert row.test_drummer not in str(row.train_drummers)
        fold = predictions.loc[
            (predictions.condition == row.condition)
            & (predictions.model == row.model)
            & (predictions.drummer == row.test_drummer)
        ]
        assert len(fold) == row.n_test
        for actual, predicted, count in [(0, 0, row.tn), (0, 1, row.fp), (1, 0, row.fn), (1, 1, row.tp)]:
            assert ((fold.y_true == actual) & (fold.y_pred == predicted)).sum() == count
    threshold_rows = metrics.loc[metrics.model == "timing_threshold"]
    assert threshold_rows.loc[threshold_rows.condition == "quantized_32_100", "threshold_feature"].eq("timing_32_abs_mean").all()
    assert threshold_rows.loc[threshold_rows.condition != "quantized_32_100", "threshold_feature"].eq("timing_abs_mean").all()
    assert threshold_rows.loc[threshold_rows.condition == "velocity_normalized", "accuracy"].eq(0.5).all()


def test_evaluate_table_can_restrict_conditions_and_feature_columns(paired_table, monkeypatch):
    sixteenth_conditions = tuple(c for c in baseline.EXPERIMENT_CONDITIONS if c.startswith("quantized_16_"))
    ablated_columns = tuple(c for c in baseline.FEATURE_COLUMNS if c not in ("timing_32_abs_mean", "timing_32_std"))

    seen_column_counts = []
    original_fit = StandardScaler.fit

    def record_width(scaler, values, *args, **kwargs):
        seen_column_counts.append(np.asarray(values).shape[1])
        return original_fit(scaler, values, *args, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_width)
    metrics, predictions = baseline.evaluate_table(
        paired_table, conditions=sixteenth_conditions, feature_columns=ablated_columns,
    )

    # Only the four 1/16 conditions ran, not all six.
    assert set(metrics.condition) == set(sixteenth_conditions)
    assert set(predictions.condition) == set(sixteenth_conditions)
    # Logistic regression only ever saw the ablated (narrower) feature set.
    assert set(seen_column_counts) == {len(ablated_columns)}
    # The timing-threshold model doesn't use FEATURE_COLUMNS at all, so
    # restricting them changes nothing about its results or behavior.
    full_metrics, _ = baseline.evaluate_table(paired_table, conditions=sixteenth_conditions)
    full_threshold = full_metrics.loc[full_metrics.model == "timing_threshold"].sort_values(["condition", "test_drummer"])
    ablated_threshold = metrics.loc[metrics.model == "timing_threshold"].sort_values(["condition", "test_drummer"])
    pd.testing.assert_frame_equal(
        full_threshold.reset_index(drop=True), ablated_threshold.reset_index(drop=True),
    )
