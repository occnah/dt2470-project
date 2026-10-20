"""Qualitative error analysis: clip metadata join, per-group error rates, examples."""

import numpy as np
import pandas as pd
import pytest

from src import error_analysis


@pytest.fixture
def features():
    """Two source clips x (human + one condition), with style/tempo metadata."""
    rows = []
    for i, (style, tempo) in enumerate([("funk", 80.0), ("rock", 140.0)]):
        for condition in ("human", "quantize_16_25"):
            rows.append({
                "source_clip_id": f"clip{i}", "condition": condition,
                "drummer": f"d{i}", "style": style, "tempo": tempo,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def predictions():
    """30 clips per group so MIN_GROUP_SIZE is cleared; group A all correct, group B all wrong."""
    rows = []
    for i in range(60):
        group = "a" if i < 30 else "b"
        source_clip_id = f"clip_{group}_{i}"
        correct = group == "a"
        rows.append({
            "condition": "quantize_16_25", "model": "logistic_regression", "drummer": "d0",
            "clip_id": f"{source_clip_id}__human", "source_clip_id": source_clip_id,
            "y_true": 0, "y_pred": 0 if correct else 1,
        })
        rows.append({
            "condition": "quantize_16_25", "model": "logistic_regression", "drummer": "d0",
            "clip_id": f"{source_clip_id}__quantize_16_25", "source_clip_id": source_clip_id,
            "y_true": 1, "y_pred": 1 if correct else 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def grouped_features():
    """60 clips split funk/rock, so error_rates has two groups of 30 to compare."""
    rows = []
    for i in range(60):
        style = "funk" if i < 30 else "rock"
        source_clip_id = f"clip_{'a' if i < 30 else 'b'}_{i}"
        for condition in ("human", "quantize_16_25"):
            rows.append({
                "source_clip_id": source_clip_id, "condition": condition,
                "drummer": "d0", "style": style, "tempo": 100.0,
            })
    return pd.DataFrame(rows)


def test_clip_metadata_keeps_only_human_rows_and_drops_drummer(features):
    meta = error_analysis.clip_metadata(features)
    assert set(meta.index) == {"clip0", "clip1"}
    assert "drummer" not in meta.columns
    assert meta.loc["clip0", "style"] == "funk"


def test_clip_metadata_buckets_tempo(features):
    meta = error_analysis.clip_metadata(features)
    assert meta.loc["clip0", "tempo_bucket"] == "<90"
    assert meta.loc["clip1", "tempo_bucket"] == "130-150"


def test_error_rates_separates_perfect_from_wrong_groups(predictions, grouped_features):
    meta = error_analysis.clip_metadata(grouped_features)
    rows = predictions.join(meta, on="source_clip_id")
    table = error_analysis.error_rates(rows, "style")
    funk = table.set_index("style").loc["funk"]
    rock = table.set_index("style").loc["rock"]
    assert funk.false_positive_rate == 0.0 and funk.false_negative_rate == 0.0
    assert rock.false_positive_rate == 1.0 and rock.false_negative_rate == 1.0


def test_error_rates_drops_groups_below_minimum_size(predictions, grouped_features):
    # First 10 funk clips (20 rows: below MIN_GROUP_SIZE on each side) plus all 30 rock clips.
    small = pd.concat([predictions.iloc[:20], predictions.iloc[60:]])
    meta = error_analysis.clip_metadata(grouped_features)
    rows = small.join(meta, on="source_clip_id")
    table = error_analysis.error_rates(rows, "style")
    assert "funk" not in set(table["style"])
    assert "rock" in set(table["style"])


def test_worst_examples_returns_only_the_requested_error_kind(predictions, grouped_features):
    meta = error_analysis.clip_metadata(grouped_features)
    rows = predictions.join(meta, on="source_clip_id")
    examples = error_analysis.worst_examples(rows, "style", "rock", "fn", n=3)
    assert len(examples) == 3
    assert set(examples["style"]) == {"rock"}


def test_write_report_runs_end_to_end(tmp_path, predictions, grouped_features):
    meta = error_analysis.clip_metadata(grouped_features)
    error_analysis.write_report(
        predictions, meta, ("quantize_16_25",), "logistic_regression", tmp_path,
    )
    text = (tmp_path / "ERROR_ANALYSIS.md").read_text(encoding="utf-8")
    assert "quantize_16_25" in text
    assert "By style" in text
    assert "rock" in text and "funk" in text


def test_write_report_handles_missing_condition(tmp_path, predictions, grouped_features):
    meta = error_analysis.clip_metadata(grouped_features)
    error_analysis.write_report(
        predictions, meta, ("quantize_16_25", "velocity_25"), "logistic_regression", tmp_path,
    )
    text = (tmp_path / "ERROR_ANALYSIS.md").read_text(encoding="utf-8")
    assert "No predictions found for this condition" in text
