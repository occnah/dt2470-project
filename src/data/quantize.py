"""
Quantization conditions.

TODO: uses a single constant tempo (get_bpm); switch to the tempo-map
grid in src.common.midi before generating the real conditions.
"""

import copy

import pretty_midi

from src.common.midi import get_bpm


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

    nearest_grid_point = round(time / grid_size) * grid_size

    return time + strength * (nearest_grid_point - time)


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

    bpm = get_bpm(midi)
    grid_size = get_grid_size_seconds(bpm, subdivision)

    for instrument in quantized.instruments:
        if not instrument.is_drum:
            continue

        for note in instrument.notes:
            original_start = note.start
            duration = note.end - note.start

            note.start = quantize_time(
                original_start,
                grid_size,
                strength,
            )

            note.end = note.start + duration

    return quantized