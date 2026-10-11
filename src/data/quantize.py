"""Tempo-aware timing quantization and independent velocity normalization."""

import copy

import numpy as np
import pretty_midi

from src.common.config import NORMALIZED_VELOCITY
from src.common.midi import nearest_grid_index, note_grid


def get_grid_size_seconds(bpm: float, subdivision: int) -> float:
    """
    Return the duration of one grid step in seconds.

    subdivision=16 -> sixteenth-note grid
    subdivision=32 -> thirty-second-note grid
    """
    quarter_note = 60.0 / bpm

    if subdivision == 16:
        return quarter_note / 4.0

    if subdivision == 32:
        return quarter_note / 8.0

    raise ValueError(f"Unsupported subdivision: {subdivision}")


def quantize_time(
    time: float,
    grid_size: float,
    strength: float = 1.0,
) -> float:
    """
    Move a time value toward the nearest grid point.

    strength:
        0.0 -> unchanged
        0.5 -> halfway toward the grid
        1.0 -> fully quantized
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0")

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    left = np.floor(time / grid_size) * grid_size
    right = left + grid_size
    earlier = time - left <= right - time or np.isclose(
        time - left, right - time, rtol=0.0, atol=1e-12
    )
    nearest_grid_point = left if earlier else right
    return time + strength * (nearest_grid_point - time)


def quantize_onsets(
    onsets: np.ndarray,
    grid: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Move onsets toward the supplied grid without changing their note order."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0")
    targets = grid[nearest_grid_index(onsets, grid)]
    if strength == 1.0:
        return targets.copy()
    return onsets + strength * (targets - onsets)


def quantize_midi(
    midi: pretty_midi.PrettyMIDI,
    subdivision: int = 16,
    strength: float = 1.0,
) -> pretty_midi.PrettyMIDI:
    """
    Return a quantized copy of a MIDI performance.

    Only drum instruments are modified.
    Note durations are preserved.
    """
    quantized = copy.deepcopy(midi)

    grid = note_grid(midi, midi.get_end_time(), subdivision)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0.0 and 1.0")

    for instrument in quantized.instruments:
        if not instrument.is_drum:
            continue

        starts = quantize_onsets(
            np.asarray([note.start for note in instrument.notes]), grid, strength
        )
        for note, start in zip(instrument.notes, starts):
            duration = note.end - note.start
            note.start = float(start)
            note.end = note.start + duration

    return quantized


def normalize_velocity_midi(
    midi: pretty_midi.PrettyMIDI,
    velocity: int = NORMALIZED_VELOCITY,
) -> pretty_midi.PrettyMIDI:
    """Return a copy with constant drum velocity; keep all note times unchanged."""
    if not isinstance(velocity, int) or not 1 <= velocity <= 127:
        raise ValueError("velocity must be an integer between 1 and 127")
    normalized = copy.deepcopy(midi)
    for instrument in normalized.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                note.velocity = velocity
    return normalized
