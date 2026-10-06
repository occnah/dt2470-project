from pathlib import Path

import pandas as pd
import pretty_midi

DEFAULT_ROOT = Path("data/groove")


def load_info(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """
    Load the Groove MIDI Dataset metadata (info.csv).

    Adds a `midi_path` column with the full path to each MIDI file.
    """
    info = pd.read_csv(root / "info.csv")
    info["midi_path"] = info["midi_filename"].map(lambda name: root / name)
    return info


def filter_info(
    info: pd.DataFrame,
    split: str | None = None,
    beat_type: str | None = None,
    time_signature: str | None = None,
) -> pd.DataFrame:
    """
    Select a subset of the dataset, e.g. split="train", beat_type="beat",
    time_signature="4-4".
    """
    if split is not None:
        info = info[info["split"] == split]

    if beat_type is not None:
        info = info[info["beat_type"] == beat_type]

    if time_signature is not None:
        info = info[info["time_signature"] == time_signature]

    return info


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
