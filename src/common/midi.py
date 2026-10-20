"""MIDI helpers: loading, drum notes, tempo-aware note grids and bar lines."""

import copy
from collections import Counter
from pathlib import Path

import mido
import numpy as np
import pretty_midi


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """Read MIDI without losing drum note-ons omitted by note-duration parsing.

    Some GMD drum hits have a note-off at the same tick as their note-on.
    Keep any omitted drum onset with its original pitch/velocity and zero
    duration. Onset-based features do not require a positive note duration.
    """
    midi = pretty_midi.PrettyMIDI(str(path))
    existing = Counter(
        (instrument.program, note.pitch, int(midi.time_to_tick(note.start)), note.velocity)
        for instrument in midi.instruments if instrument.is_drum
        for note in instrument.notes
    )
    drums = {}
    for instrument in midi.instruments:
        if instrument.is_drum:
            drums.setdefault(instrument.program, instrument)
    for track in mido.MidiFile(str(path)).tracks:
        tick, program = 0, 0
        for event in track:
            tick += event.time
            if event.type == "program_change" and event.channel == 9:
                program = event.program
            if event.type != "note_on" or event.channel != 9 or event.velocity == 0:
                continue
            key = (program, event.note, tick, event.velocity)
            if existing[key]:
                existing[key] -= 1
                continue
            if program not in drums:
                drums[program] = pretty_midi.Instrument(program, is_drum=True)
                midi.instruments.append(drums[program])
            start = midi.tick_to_time(tick)
            drums[program].notes.append(pretty_midi.Note(event.velocity, event.note, start, start))
    return midi


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


# GMD (Roland TD-11) pitches grouped as in the GMD paper. Everything else
# (toms, cymbals) is "other".
DRUM_CLASSES = {
    36: "kick",
    37: "snare", 38: "snare", 40: "snare",
    22: "hihat", 26: "hihat", 42: "hihat", 44: "hihat", 46: "hihat",
}


def drum_classes(pitches: np.ndarray) -> np.ndarray:
    """Map MIDI pitches to "kick", "snare", "hihat" or "other"."""
    return np.asarray([DRUM_CLASSES.get(int(pitch), "other") for pitch in pitches])


def with_drum_notes(
    midi: pretty_midi.PrettyMIDI,
    onsets: np.ndarray,
    velocities: np.ndarray,
) -> pretty_midi.PrettyMIDI:
    """
    Return a copy of `midi` whose drum notes have the given onsets and
    velocities, in the order returned by `get_drum_notes`.

    Note durations, pitches and non-drum instruments are kept. The input
    is not modified.
    """
    result = copy.deepcopy(midi)
    notes = [
        note
        for instrument in result.instruments
        if instrument.is_drum
        for note in instrument.notes
    ]
    notes.sort(key=lambda note: (note.start, note.pitch))
    if len(notes) != len(onsets) or len(notes) != len(velocities):
        raise ValueError("onsets and velocities must match the drum notes")

    for note, onset, velocity in zip(notes, onsets, velocities):
        duration = note.end - note.start
        note.start = max(float(onset), 0.0)
        note.end = note.start + duration
        note.velocity = int(round(velocity))

    return result


def note_grid(
    midi: pretty_midi.PrettyMIDI,
    end_time: float,
    subdivision: int = 16,
) -> np.ndarray:
    """
    Return a 1/16 or 1/32 grid from time zero to just past `end_time`.

    Grid points are placed in MIDI ticks and converted to seconds through
    the file's tempo map, so the grid follows the MIDI tempo rather than
    a fixed BPM.
    """
    if subdivision not in (16, 32):
        raise ValueError(f"Unsupported subdivision: {subdivision}")
    steps_per_quarter = subdivision // 4
    if midi.resolution % steps_per_quarter:
        raise ValueError("MIDI resolution cannot represent the requested grid exactly")
    step = midi.resolution // steps_per_quarter
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
    Index of the nearest grid point for each onset. Ties go to the earlier
    point, allowing 1e-12 seconds for tick-to-seconds rounding.
    """
    right = np.clip(np.searchsorted(grid, onsets), 1, len(grid) - 1)
    left = right - 1

    left_distance = onsets - grid[left]
    right_distance = grid[right] - onsets
    ties = np.isclose(left_distance, right_distance, rtol=0.0, atol=1e-12)
    return np.where((left_distance <= right_distance) | ties, left, right)


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
