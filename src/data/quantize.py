"""
Quantization: move onsets toward a grid, and velocities toward a constant.

Both work on plain arrays and return new arrays; inputs are not modified.
"""

import numpy as np

from src.common.midi import nearest_grid_index


def _check_strength(strength: float) -> None:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0")


def quantize_onsets(
    onsets: np.ndarray,
    grid: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Move each onset `strength` of the way toward its nearest grid point:
    new = onset + strength * (grid point - onset).

    strength 0 keeps the onset, 1 puts it exactly on the grid. `grid` is a
    tempo-aware grid from `note_grid`.
    """
    _check_strength(strength)
    targets = grid[nearest_grid_index(onsets, grid)]
    if strength == 1.0:
        return targets.copy()
    return onsets + strength * (targets - onsets)


def normalize_velocities(
    velocities: np.ndarray,
    target: float,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Move each velocity `strength` of the way toward a constant `target`:
    new = velocity + strength * (target - velocity).

    strength 1 gives every note the target velocity (DAW-style velocity
    normalization); smaller strengths compress the dynamics partially.
    """
    _check_strength(strength)
    if not 1 <= target <= 127:
        raise ValueError("target velocity must be between 1 and 127")
    if strength == 1.0:
        return np.full_like(velocities, target, dtype=float)
    return velocities + strength * (target - velocities)
