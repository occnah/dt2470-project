"""MIDI helpers: drum-note extraction, tempo and the tempo-aware 1/16 grid."""

import numpy as np
import pretty_midi


def get_drum_notes(
    midi: pretty_midi.PrettyMIDI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (onsets, pitches, velocities) of all drum notes, sorted by onset.

    Timing and velocity are returned exactly as stored in the MIDI file.
    """
    notes = [
        note
        for instrument in midi.instruments
        if instrument.is_drum
        for note in instrument.notes
    ]
    notes.sort(key=lambda note: (note.start, note.pitch))

    onsets = np.asarray([note.start for note in notes], dtype=float)
    pitches = np.asarray([note.pitch for note in notes], dtype=int)
    velocities = np.asarray([note.velocity for note in notes], dtype=float)

    return onsets, pitches, velocities


def get_bpm(midi: pretty_midi.PrettyMIDI) -> float:
    """
    Return the tempo stored in the MIDI file.

    Groove MIDI files contain exactly one tempo event, which is the click
    tempo the drummer played to. This is far more reliable than
    `estimate_tempo()`, which typically returns double the real tempo.
    """
    _, tempi = midi.get_tempo_changes()

    if len(tempi) != 1:
        raise ValueError(f"Expected a single tempo, found {len(tempi)}")

    return float(tempi[0])


def sixteenth_grid(
    midi: pretty_midi.PrettyMIDI,
    end_time: float,
) -> np.ndarray:
    """
    Return the times (seconds) of every 1/16 note from the start of the
    file to just past `end_time`.

    Grid points are placed in MIDI ticks and converted to seconds through
    the file's tempo map, so the grid follows the MIDI tempo rather than
    a fixed BPM.
    """
    step = midi.resolution // 4
    last_step = midi.time_to_tick(end_time) // step + 2

    return np.asarray(
        [midi.tick_to_time(k * step) for k in range(last_step + 1)],
        dtype=float,
    )


def nearest_grid_index(
    onsets: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Index of the nearest grid point for each onset. Exact ties go to the
    earlier grid point.
    """
    right = np.clip(np.searchsorted(grid, onsets), 1, len(grid) - 1)
    left = right - 1

    return np.where(onsets - grid[left] <= grid[right] - onsets, left, right)


def downbeat_ticks(
    midi: pretty_midi.PrettyMIDI,
    until_tick: int,
) -> list[int]:
    """
    Return the tick of every bar line from the start of the file up to and
    including `until_tick`, following the file's time signature(s).

    Unlike `PrettyMIDI.get_downbeats()`, this is not limited to the MIDI
    end time, so the bar line closing the final bar is included.
    """
    changes = midi.time_signature_changes
    ticks = []

    for i, ts in enumerate(changes):
        start = int(midi.time_to_tick(ts.time))
        stop = (
            int(midi.time_to_tick(changes[i + 1].time))
            if i + 1 < len(changes)
            else until_tick + 1
        )
        bar = ts.numerator * midi.resolution * 4 // ts.denominator
        ticks.extend(range(start, stop, bar))

    return ticks
