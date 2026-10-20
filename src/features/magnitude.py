"""
Magnitude features (`mag_*`): how large the timing, IOI and velocity
deviations are, ignoring how they are arranged.

Times are in seconds; standard deviations use ddof=0.
"""

import numpy as np

from src.common.config import IOI_EVENT_TOLERANCE
from src.common.midi import nearest_grid_index

INSTRUMENTS = ("kick", "snare", "hihat")


def grid_deviations(onsets: np.ndarray, grid: np.ndarray) -> np.ndarray:
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
    Merge onsets within `tolerance` seconds of their neighbour into one
    event and return each event's mean time. `onsets` need not be sorted.

    Keeps the number of IOI events comparable across conditions: a kick and
    hi-hat struck a few ms apart stay one event when quantization pulls
    them exactly together.
    """
    if len(onsets) == 0:
        return np.asarray(onsets, dtype=float)
    ordered = np.sort(onsets)
    groups = [[ordered[0]]]
    for time in ordered[1:]:
        if time - groups[-1][-1] <= tolerance:
            groups[-1].append(time)
        else:
            groups.append([time])
    return np.array([np.mean(group) for group in groups])


def magnitude_features(
    onsets: np.ndarray,
    velocities: np.ndarray,
    instruments: np.ndarray,
    grid_16: np.ndarray,
    grid_32: np.ndarray,
) -> dict[str, float]:
    """
    Magnitude features for one clip.

    - timing: mean absolute and SD of deviations from the 1/16 and the 1/32
      grid, and the mean absolute 1/16 deviation per instrument (NaN when
      the instrument is not played in the clip)
    - IOI: mean and SD of intervals between onset events (NaN with fewer
      than two events)
    - velocity: mean, SD and range
    """
    deviations = grid_deviations(onsets, grid_16)
    deviations_32 = grid_deviations(onsets, grid_32)
    iois = np.diff(group_onset_events(onsets))

    features = {
        "mag_timing_abs_mean": float(np.mean(np.abs(deviations))),
        "mag_timing_std": float(np.std(deviations)),
        "mag_timing32_abs_mean": float(np.mean(np.abs(deviations_32))),
        "mag_timing32_std": float(np.std(deviations_32)),
    }
    for instrument in INSTRUMENTS:
        played = instruments == instrument
        features[f"mag_{instrument}_timing_abs_mean"] = (
            float(np.mean(np.abs(deviations[played]))) if played.any() else float("nan")
        )
    features.update({
        "mag_ioi_mean": float(np.mean(iois)) if len(iois) else float("nan"),
        "mag_ioi_std": float(np.std(iois)) if len(iois) else float("nan"),
        "mag_velocity_mean": float(np.mean(velocities)),
        "mag_velocity_std": float(np.std(velocities)),
        "mag_velocity_range": float(np.ptp(velocities)),
    })
    return features
