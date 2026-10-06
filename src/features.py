import numpy as np
import pretty_midi

from dataset import get_bpm
from quantize import get_grid_size_seconds


def get_drum_notes(
    midi: pretty_midi.PrettyMIDI,
) -> list[pretty_midi.Note]:
    """
    Return all drum notes sorted by onset time.
    """
    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            notes.extend(instrument.notes)

    return sorted(notes, key=lambda note: note.start)


def timing_deviations(
    midi: pretty_midi.PrettyMIDI,
    subdivision: int = 16,
) -> np.ndarray:
    """
    Calculate onset deviation from the nearest rhythmic grid point.
    """
    bpm = get_bpm(midi)
    grid_size = get_grid_size_seconds(bpm, subdivision)

    deviations = []

    for note in get_drum_notes(midi):
        nearest = round(note.start / grid_size) * grid_size
        deviations.append(note.start - nearest)

    return np.asarray(deviations)


def extract_features(
    midi: pretty_midi.PrettyMIDI,
) -> dict[str, float]:
    """
    Extract an initial small set of rhythmic features.
    """
    notes = get_drum_notes(midi)

    if not notes:
        raise ValueError("No drum notes found in MIDI file")

    deviations = timing_deviations(midi, subdivision=16)

    velocities = np.asarray(
        [note.velocity for note in notes],
        dtype=float,
    )

    onsets = np.asarray(
        [note.start for note in notes],
        dtype=float,
    )

    iois = np.diff(onsets)

    return {
        "note_count": len(notes),
        "timing_abs_mean": float(np.mean(np.abs(deviations))),
        "timing_std": float(np.std(deviations)),
        "timing_max_abs": float(np.max(np.abs(deviations))),
        "velocity_mean": float(np.mean(velocities)),
        "velocity_std": float(np.std(velocities)),
        "velocity_range": float(np.ptp(velocities)),
        "ioi_mean": float(np.mean(iois)) if len(iois) else 0.0,
        "ioi_std": float(np.std(iois)) if len(iois) else 0.0,
    }