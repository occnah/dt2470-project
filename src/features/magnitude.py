"""
Magnitude features: the size of timing, IOI and velocity deviations.

Times are in seconds; standard deviations use ddof=0.
"""

import numpy as np

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


def clip_features(
    onsets: np.ndarray,
    velocities: np.ndarray,
    grid: np.ndarray,
) -> dict[str, float]:
    """
    Rhythmic features for one clip.

    IOIs are taken between distinct onset times, so notes stored at the
    same instant (e.g. kick + hi-hat) count as one event. IOI features
    are NaN if the clip has fewer than two distinct onsets.
    """
    deviations = grid_deviations(onsets, grid)
    iois = np.diff(np.unique(onsets))

    return {
        "note_count": len(onsets),
        "timing_abs_mean": float(np.mean(np.abs(deviations))),
        "timing_std": float(np.std(deviations)),
        "ioi_mean": float(np.mean(iois)) if len(iois) else float("nan"),
        "ioi_std": float(np.std(iois)) if len(iois) else float("nan"),
        "velocity_mean": float(np.mean(velocities)),
        "velocity_std": float(np.std(velocities)),
        "velocity_range": float(np.ptp(velocities)),
    }
