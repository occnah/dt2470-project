"""Groove MIDI Dataset metadata (info.csv) and the dataset scope filter."""

from pathlib import Path

import pandas as pd

from src.common.config import EXCLUDED_STYLE_KEYWORDS, VALID_BEAT_TYPE, VALID_TIME_SIGNATURE


def load_metadata(root: Path) -> dict[str, dict]:
    """
    Return GMD metadata keyed by MIDI path relative to `root`
    (e.g. "drummer1/session1/1_funk_80_beat_4-4.mid"), or {} if `root`
    has no info.csv.
    """
    path = root / "info.csv"
    if not path.exists():
        return {}
    return pd.read_csv(path).set_index("midi_filename").to_dict("index")


def scope_exclusion(meta: dict) -> str | None:
    """
    Why a performance is outside the experiment's scope, or None if it is
    in scope: an ordinary 4/4 groove whose style is not labelled as swing,
    shuffle or triplet (see `src/common/config.py`).
    """
    if meta.get("beat_type") != VALID_BEAT_TYPE:
        return "not a beat (fill)"
    if meta.get("time_signature") != VALID_TIME_SIGNATURE:
        return "not 4/4"
    style = str(meta.get("style", "")).lower()
    if any(keyword in style for keyword in EXCLUDED_STYLE_KEYWORDS):
        return "swing/shuffle/triplet style"
    return None
