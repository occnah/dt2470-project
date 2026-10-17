"""
Magnitude features: the size of timing, IOI and velocity deviations.

Times are in seconds; standard deviations use ddof=0.
"""

import numpy as np

from src.common.config import IOI_EVENT_TOLERANCE
from src.common.midi import nearest_grid_index


def grid_deviations(
    onsets: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Signed distance (seconds) from each onset to its nearest grid point.
    Positive means the note was played late.
    """
    return onsets - grid[nearest_grid_index(onsets, grid)]


def group_onset_events(
    onsets: np.ndarray,
    tolerance: float = IOI_EVENT_TOLERANCE,
) -> np.ndarray:
    """
    Merge onsets within `tolerance` seconds of their neighbor into one
    event, returning one representative time (the group's mean) per
    event. `onsets` need not be sorted or unique.

    This keeps the number of IOI events comparable across conditions:
    quantization can pull two near-simultaneous onsets (e.g. a kick and
    hi-hat struck a few milliseconds apart) exactly on top of each other,
    which would otherwise change how many distinct events exist and let
    IOI features partly detect quantization through that counting
    artifact rather than through genuine timing deviation. The same
    tolerance is applied to every condition.
    """
    if len(onsets) == 0:
        return np.asarray(onsets, dtype=float)
    sorted_onsets = np.sort(onsets)
    groups = [[sorted_onsets[0]]]
    for time in sorted_onsets[1:]:
        if time - groups[-1][-1] <= tolerance:
            groups[-1].append(time)
        else:
            groups.append([time])
    return np.array([np.mean(group) for group in groups])


def clip_features(
    onsets: np.ndarray,
    velocities: np.ndarray,
    grid: np.ndarray,
    grid_32: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Rhythmic features for one clip.

    IOIs are taken between onset events (see `group_onset_events`), so
    notes within `IOI_EVENT_TOLERANCE` seconds of each other (e.g. kick +
    hi-hat struck together) count as one event. IOI features are NaN if
    the clip has fewer than two distinct onset events.
    """
    deviations = grid_deviations(onsets, grid)
    iois = np.diff(group_onset_events(onsets))

    features = {
        "note_count": len(onsets),
        "timing_abs_mean": float(np.mean(np.abs(deviations))),
        "timing_std": float(np.std(deviations)),
        "ioi_mean": float(np.mean(iois)) if len(iois) else float("nan"),
        "ioi_std": float(np.std(iois)) if len(iois) else float("nan"),
        "velocity_mean": float(np.mean(velocities)),
        "velocity_std": float(np.std(velocities)),
        "velocity_range": float(np.ptp(velocities)),
    }
    if grid_32 is not None:
        deviations_32 = grid_deviations(onsets, grid_32)
        features["timing_32_abs_mean"] = float(np.mean(np.abs(deviations_32)))
        features["timing_32_std"] = float(np.std(deviations_32))
    return features
