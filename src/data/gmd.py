"""Groove MIDI Dataset metadata (info.csv) loading and filtering."""

from pathlib import Path

import pandas as pd

from src.common.config import RAW_DIR


def load_info(root: Path = RAW_DIR) -> pd.DataFrame:
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
