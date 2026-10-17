import math

import numpy as np
import pytest

from src.common.midi import nearest_grid_index
from src.features.magnitude import clip_features, grid_deviations, group_onset_events

GRID = np.array([0.0, 0.125, 0.25, 0.375, 0.5])


def test_nearest_grid_index():
    onsets = np.array([0.0, 0.01, 0.12, 0.2, 0.49, 0.6])
    assert nearest_grid_index(onsets, GRID).tolist() == [0, 0, 1, 2, 4, 4]


def test_nearest_grid_index_tie_goes_to_earlier_point():
    assert nearest_grid_index(np.array([0.0625]), GRID).tolist() == [0]


def test_grid_deviations_are_signed():
    onsets = np.array([0.01, 0.115, 0.25])
    np.testing.assert_allclose(grid_deviations(onsets, GRID), [0.01, -0.01, 0.0])


def test_grid_deviations_on_uneven_grid():
    # A grid that slows down halfway, as after a tempo change.
    grid = np.array([0.0, 0.1, 0.2, 0.4, 0.6])
    np.testing.assert_allclose(grid_deviations(np.array([0.21, 0.39]), grid), [0.01, -0.01])


def test_clip_features_values():
    onsets = np.array([0.01, 0.125, 0.125, 0.24, 0.5])
    velocities = np.array([40.0, 60.0, 60.0, 100.0, 80.0])
    features = clip_features(onsets, velocities, GRID)

    deviations = np.array([0.01, 0.0, 0.0, -0.01, 0.0])
    iois = np.array([0.115, 0.115, 0.26])

    assert features["note_count"] == 5
    assert features["timing_abs_mean"] == pytest.approx(0.004)
    assert features["timing_std"] == pytest.approx(np.std(deviations))
    assert features["ioi_mean"] == pytest.approx(iois.mean())
    assert features["ioi_std"] == pytest.approx(iois.std())
    assert features["velocity_mean"] == pytest.approx(68.0)
    assert features["velocity_std"] == pytest.approx(velocities.std())
    assert features["velocity_range"] == pytest.approx(60.0)


def test_clip_features_on_grid_is_zero():
    features = clip_features(GRID.copy(), np.full(5, 90.0), GRID)
    assert features["timing_abs_mean"] == 0.0
    assert features["timing_std"] == 0.0
    assert features["velocity_std"] == 0.0


def test_clip_features_single_onset_has_nan_ioi():
    features = clip_features(np.array([0.25, 0.25]), np.array([80.0, 90.0]), GRID)
    assert math.isnan(features["ioi_mean"])
    assert math.isnan(features["ioi_std"])


def test_group_onset_events_merges_within_tolerance():
    # 1.000 and 1.012 are a near-simultaneous kick+hi-hat (within 20ms);
    # 1.247 and 1.254 are another such pair. Quantization could otherwise
    # pull either pair exactly together and silently drop an IOI event.
    onsets = np.array([1.000, 1.012, 1.247, 1.254])
    events = group_onset_events(onsets, tolerance=0.02)
    np.testing.assert_allclose(events, [1.006, 1.2505])


def test_group_onset_events_keeps_events_past_tolerance_apart():
    onsets = np.array([1.000, 1.025])  # 25ms apart: two distinct events
    events = group_onset_events(onsets, tolerance=0.02)
    np.testing.assert_allclose(events, [1.000, 1.025])


def test_group_onset_events_is_order_independent():
    onsets = np.array([1.254, 1.000, 1.012, 1.247])
    np.testing.assert_allclose(
        group_onset_events(onsets, tolerance=0.02),
        group_onset_events(np.sort(onsets), tolerance=0.02),
    )


def test_clip_features_ioi_uses_grouped_events_not_raw_uniques():
    # Two near-simultaneous onset pairs 20ms apart within each pair, but
    # the pairs themselves are not exact duplicates, so np.unique would
    # treat this as four events and one IOI would be ~0. Grouping treats
    # each pair as a single event.
    onsets = np.array([1.000, 1.012, 1.247, 1.254])
    velocities = np.full(4, 80.0)
    grid = np.array([0.0, 1.0, 1.25, 1.5])
    features = clip_features(onsets, velocities, grid)
    assert features["ioi_mean"] == pytest.approx(1.2505 - 1.006)
    assert features["ioi_std"] == pytest.approx(0.0)
