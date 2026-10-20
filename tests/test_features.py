"""Magnitude and structure features on small hand-made clips."""

import math

import numpy as np
import pytest

from src.common.midi import drum_classes, nearest_grid_index
from src.features import (
    CONTEXT_FEATURES, FEATURE_COLUMNS, FEATURE_GROUPS, MODEL_INPUTS, OPTIONAL_FEATURES, extract_features,
)
from src.features.magnitude import grid_deviations, group_onset_events, magnitude_features
from src.features.structure import (
    chord_spread_ratio, explained_variance, lag1_autocorrelation, structure_features,
)

GRID = np.arange(0, 2.0001, 0.125)  # 1/16 notes at 120 BPM
GRID_32 = np.arange(0, 2.0001, 0.0625)


# --- Grid and IOI helpers --------------------------------------------------

def test_nearest_grid_index_and_ties():
    onsets = np.array([0.0, 0.01, 0.12, 0.2, 0.0625])
    assert nearest_grid_index(onsets, GRID).tolist() == [0, 0, 1, 2, 0]


def test_grid_deviations_are_signed_and_follow_an_uneven_grid():
    np.testing.assert_allclose(grid_deviations(np.array([0.01, 0.115, 0.25]), GRID), [0.01, -0.01, 0.0])
    uneven = np.array([0.0, 0.1, 0.2, 0.4, 0.6])  # slows down, as after a tempo change
    np.testing.assert_allclose(grid_deviations(np.array([0.21, 0.39]), uneven), [0.01, -0.01])


def test_onsets_within_the_tolerance_form_one_event():
    np.testing.assert_allclose(group_onset_events(np.array([1.0, 1.012, 1.247, 1.254]), 0.02), [1.006, 1.2505])
    np.testing.assert_allclose(group_onset_events(np.array([1.0, 1.025]), 0.02), [1.0, 1.025])
    np.testing.assert_allclose(group_onset_events(np.array([1.254, 1.0, 1.012, 1.247]), 0.02), [1.006, 1.2505])


# --- Magnitude features ----------------------------------------------------

def test_magnitude_features_values():
    onsets = np.array([0.01, 0.125, 0.125, 0.24, 0.5])
    velocities = np.array([40.0, 60.0, 60.0, 100.0, 80.0])
    instruments = np.array(["kick", "hihat", "snare", "hihat", "kick"])
    features = magnitude_features(onsets, velocities, instruments, GRID, GRID_32)
    deviations = np.array([0.01, 0.0, 0.0, -0.01, 0.0])
    events = group_onset_events(onsets)
    assert features["mag_timing_abs_mean"] == pytest.approx(0.004)
    assert features["mag_timing_std"] == pytest.approx(np.std(deviations))
    assert features["mag_kick_timing_abs_mean"] == pytest.approx(0.005)
    assert features["mag_hihat_timing_abs_mean"] == pytest.approx(0.005)
    assert features["mag_snare_timing_abs_mean"] == pytest.approx(0.0)
    assert features["mag_ioi_mean"] == pytest.approx(np.diff(events).mean())
    assert features["mag_velocity_mean"] == pytest.approx(68.0)
    assert features["mag_velocity_range"] == pytest.approx(60.0)


def test_thirty_second_features_use_the_finer_grid():
    features = magnitude_features(np.array([0.0625, 0.1875]), np.array([70.0, 90.0]),
                                  np.array(["kick", "kick"]), GRID, GRID_32)
    assert features["mag_timing_abs_mean"] == 0.0625
    assert features["mag_timing32_abs_mean"] == features["mag_timing32_std"] == 0.0


def test_undefined_magnitude_features_are_nan():
    features = magnitude_features(np.array([0.25, 0.25]), np.array([80.0, 90.0]),
                                  np.array(["kick", "kick"]), GRID, GRID_32)
    assert math.isnan(features["mag_ioi_mean"]) and math.isnan(features["mag_snare_timing_abs_mean"])


# --- Structure features ----------------------------------------------------

def test_lag1_autocorrelation():
    assert lag1_autocorrelation(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) > 0.3
    assert lag1_autocorrelation(np.array([1.0, -1.0, 1.0, -1.0, 1.0])) < -0.5
    assert lag1_autocorrelation(np.array([1.0, 2.0])) == 0.0
    assert lag1_autocorrelation(np.zeros(5)) == 0.0


def test_explained_variance():
    values = np.array([1.0, 1.0, 3.0, 3.0])
    assert explained_variance(values, np.array(["a", "a", "b", "b"])) == pytest.approx(1.0)
    assert explained_variance(values, np.array(["a", "b", "a", "b"])) == pytest.approx(0.0)
    assert explained_variance(np.ones(4), np.array(["a", "a", "b", "b"])) == 0.0


def test_chord_spread_counts_one_hit_per_instrument():
    deviations = np.array([0.010, 0.012, -0.030, 0.030, 0.0])
    steps = np.array([0, 0, 1, 1, 2])
    instruments = np.array(["kick", "hihat", "snare", "snare", "kick"])
    # Only step 0 has two different instruments; the two snare hits on step 1 are one drum.
    pooled = np.sqrt(np.sum((deviations[:2] - deviations[:2].mean()) ** 2) / 1)
    assert chord_spread_ratio(deviations, steps, instruments) == pytest.approx(pooled / np.std(deviations))
    assert math.isnan(chord_spread_ratio(deviations[2:4], steps[2:4], instruments[2:4]))


def test_structure_features_are_scale_free():
    onsets = np.array([0.0, 0.135, 0.25, 0.39, 0.5, 0.63, 0.75, 0.885, 0.0])
    instruments = np.array(["hihat"] * 8 + ["kick"])
    velocities = np.array([90.0, 60, 85, 55, 95, 60, 80, 50, 100])
    steps = nearest_grid_index(onsets, GRID)
    deviations = onsets - GRID[steps]
    scaled = GRID[steps] + 0.4 * deviations
    original = structure_features(onsets, velocities, instruments, GRID)
    for name, value in structure_features(scaled, 0.5 * velocities + 40, instruments, GRID).items():
        assert value == pytest.approx(original[name], nan_ok=True)
    # Late off-beats: deviation is fully explained by position in the beat.
    assert original["struct_timing_position_r2"] > 0.9


def test_a_clip_without_timing_spread_has_zero_timing_structure():
    features = structure_features(GRID[:8].copy(), np.full(8, 80.0), np.array(["hihat"] * 8), GRID)
    assert all(value == 0.0 for value in features.values())


# --- Feature registry --------------------------------------------------------

def test_every_extracted_feature_is_in_exactly_one_group():
    features = extract_features(np.array([0.01, 0.13, 0.25, 0.26]), np.array([80.0, 70, 90, 60]),
                                np.array([36, 42, 38, 42]), GRID, GRID_32)
    grouped = [name for columns in FEATURE_GROUPS.values() for name in columns]
    assert sorted(features) == sorted(grouped) == sorted(FEATURE_COLUMNS)
    assert len(grouped) == len(set(grouped))
    assert MODEL_INPUTS == CONTEXT_FEATURES + FEATURE_COLUMNS
    assert set(OPTIONAL_FEATURES) <= set(FEATURE_COLUMNS)
    assert all(name.startswith(("mag_", "struct_")) for name in FEATURE_COLUMNS)


def test_drum_classes():
    assert drum_classes(np.array([36, 38, 40, 42, 44, 49])).tolist() == [
        "kick", "snare", "snare", "hihat", "hihat", "other",
    ]
