"""
Structure features (`struct_*`): how timing and velocity deviations are
arranged, independent of their size.

Every feature here is scale-free: multiplying all deviations by a constant
does not change it. Partial quantization does exactly that, so these
features cannot detect it by construction; they are meant to separate human
timing from random jitter (RQ2, RQ3).

- struct_timing_autocorr       lag-1 autocorrelation of deviations over
                               successive hits (correlation over time)
- struct_timing_position_r2    share of deviation variance explained by the
                               position in the beat (metric position)
- struct_timing_instrument_r2  share explained by the instrument
                               (consistent per-instrument offsets)
- struct_chord_spread_ratio    spread between different instruments struck
                               on the same grid point, relative to the
                               overall spread (coordination between
                               instruments; NaN without such hits)
- struct_velocity_position_r2  share of velocity variance explained by the
                               position in the beat (accent pattern)

A clip with (almost) no timing spread, such as a fully quantized one, has
no timing structure; its timing features are 0 by convention
(`MIN_TIMING_SPREAD`), in every condition.
"""

import numpy as np

from src.common.config import MIN_TIMING_SPREAD
from src.common.midi import nearest_grid_index

STEPS_PER_BEAT = 4  # 1/16 notes per quarter note


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Lag-1 autocorrelation; 0 with fewer than 3 values or no variance."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return 0.0
    centred = values - values.mean()
    denominator = np.sum(centred ** 2)
    if denominator <= 0:
        return 0.0
    return float(np.sum(centred[:-1] * centred[1:]) / denominator)


def explained_variance(values: np.ndarray, groups: np.ndarray) -> float:
    """
    Share of the variance of `values` explained by `groups` (eta squared):
    between-group sum of squares over total sum of squares. 0 if the values
    have no variance.
    """
    values = np.asarray(values, dtype=float)
    total = np.sum((values - values.mean()) ** 2)
    if total <= 0:
        return 0.0
    between = 0.0
    for group in np.unique(groups):
        members = values[groups == group]
        between += len(members) * (members.mean() - values.mean()) ** 2
    return float(between / total)


def chord_spread_ratio(
    deviations: np.ndarray,
    steps: np.ndarray,
    instruments: np.ndarray,
) -> float:
    """
    How tightly different instruments struck on the same grid point line up.

    For every grid point played by two or more instruments, take each
    instrument's hit closest to the grid point (so drags and ghost notes on
    one drum do not count) and measure their spread. Returns the pooled SD
    of those hits divided by the clip's overall deviation SD: near 0 when
    instruments meant together land together, near 1 when they are as
    spread out as any other hits. NaN if no grid point has two instruments.
    """
    within, count = 0.0, 0
    for step in np.unique(steps):
        at_step = steps == step
        hits = []
        for instrument in np.unique(instruments[at_step]):
            candidates = deviations[at_step & (instruments == instrument)]
            hits.append(candidates[np.argmin(np.abs(candidates))])
        if len(hits) >= 2:
            hits = np.asarray(hits)
            within += np.sum((hits - hits.mean()) ** 2)
            count += len(hits) - 1
    if count == 0:
        return float("nan")
    return float(np.sqrt(within / count) / np.std(deviations))


def structure_features(
    onsets: np.ndarray,
    velocities: np.ndarray,
    instruments: np.ndarray,
    grid_16: np.ndarray,
) -> dict[str, float]:
    """Structure features for one clip (see module docstring)."""
    steps = nearest_grid_index(onsets, grid_16)
    deviations = onsets - grid_16[steps]
    positions = steps % STEPS_PER_BEAT

    features = {"struct_velocity_position_r2": explained_variance(velocities, positions)}

    if np.std(deviations) < MIN_TIMING_SPREAD:
        features.update({
            "struct_timing_autocorr": 0.0,
            "struct_timing_position_r2": 0.0,
            "struct_timing_instrument_r2": 0.0,
            "struct_chord_spread_ratio": 0.0,
        })
        return features

    # One value per grid point played, in time order: hits on the same grid
    # point (e.g. kick + hi-hat) are averaged into one event.
    event_steps = np.unique(steps)
    event_deviations = np.array([deviations[steps == step].mean() for step in event_steps])

    features.update({
        "struct_timing_autocorr": lag1_autocorrelation(event_deviations),
        "struct_timing_position_r2": explained_variance(deviations, positions),
        "struct_timing_instrument_r2": explained_variance(deviations, instruments),
        "struct_chord_spread_ratio": chord_spread_ratio(deviations, steps, instruments),
    })
    return features
