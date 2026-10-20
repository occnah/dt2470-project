"""
Features: per-clip measurements used as classifier inputs.

    magnitude.py  `mag_*`    how large deviations are
    structure.py  `struct_*` how deviations are arranged (scale-free)

`FEATURE_GROUPS` is the single list of model features, grouped for the
ablation in `src/run_ablation.py`. `CONTEXT_FEATURES` describe the clip
itself (tempo, number of notes); they are identical in every version of a
clip, so they carry no information about the condition on their own, but
let models interpret the other features (e.g. seconds at a given tempo).
"""

import numpy as np

from src.common.midi import drum_classes
from src.features.magnitude import INSTRUMENTS, magnitude_features
from src.features.structure import structure_features

CONTEXT_FEATURES = ("tempo", "note_count")

FEATURE_GROUPS = {
    "mag_timing": (
        "mag_timing_abs_mean", "mag_timing_std",
        "mag_timing32_abs_mean", "mag_timing32_std",
        *(f"mag_{instrument}_timing_abs_mean" for instrument in INSTRUMENTS),
    ),
    "mag_ioi": ("mag_ioi_mean", "mag_ioi_std"),
    "mag_velocity": ("mag_velocity_mean", "mag_velocity_std", "mag_velocity_range"),
    "struct_timing": (
        "struct_timing_autocorr", "struct_timing_position_r2",
        "struct_timing_instrument_r2", "struct_chord_spread_ratio",
    ),
    "struct_velocity": ("struct_velocity_position_r2",),
}

FEATURE_COLUMNS = tuple(name for group in FEATURE_GROUPS.values() for name in group)
MAGNITUDE_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c.startswith("mag_"))
STRUCTURE_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c.startswith("struct_"))
MODEL_INPUTS = CONTEXT_FEATURES + FEATURE_COLUMNS

# Features that are undefined (NaN) for some clips: an instrument that is
# not played, a single onset event, or no simultaneous hits. This depends
# almost only on the notes, which every version of a clip shares (jitter can
# occasionally merge or split IOI events or shared grid points). Models fill
# these values inside each training fold (see src/models/classifiers.py).
OPTIONAL_FEATURES = (
    *(f"mag_{instrument}_timing_abs_mean" for instrument in INSTRUMENTS),
    "mag_ioi_mean", "mag_ioi_std", "struct_chord_spread_ratio",
)


def extract_features(
    onsets: np.ndarray,
    velocities: np.ndarray,
    pitches: np.ndarray,
    grid_16: np.ndarray,
    grid_32: np.ndarray,
) -> dict[str, float]:
    """All `mag_*` and `struct_*` features for one clip's notes."""
    instruments = drum_classes(pitches)
    return {
        **magnitude_features(onsets, velocities, instruments, grid_16, grid_32),
        **structure_features(onsets, velocities, instruments, grid_16),
    }
