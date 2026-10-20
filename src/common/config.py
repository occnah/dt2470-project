"""
Project-wide constants: paths, seed, dataset scope and analysis settings.

Conditions live in `src/data/conditions.py`, features in `src/features/`,
and models in `src/models/classifiers.py`.
"""

from pathlib import Path

SEED = 42
BARS_PER_CLIP = 2

# --- Paths ---------------------------------------------------------------

# The Groove MIDI Dataset as downloaded (contains info.csv).
RAW_DIR = Path("data/groove")
# Generated feature tables and predictions (not in git).
FEATURES_DIR = Path("data/features")
FEATURES_TABLE = FEATURES_DIR / "features.csv"
# Small, shareable summaries and figures (tracked in git).
RESULTS_DIR = Path("results")

# --- Dataset scope -------------------------------------------------------

# Ordinary 4/4 grooves only. Fills are excluded, and so are performances
# whose GMD style label marks a swing, shuffle or triplet feel: forcing
# those onto a straight 1/16 grid would be detectable for reasons
# unrelated to quantization ("swing vs straight"). Onset positions alone
# do not separate swung from straight GMD files reliably, so the drummers'
# own style labels are used. Unlabelled styles such as plain "jazz" stay in.
VALID_BEAT_TYPE = "beat"
VALID_TIME_SIGNATURE = "4-4"
EXCLUDED_STYLE_KEYWORDS = ("swing", "shuffle", "triplet")

# --- Features ------------------------------------------------------------

# Onsets within this many seconds of each other count as one event for
# inter-onset intervals (IOIs), identically in every condition. Without
# this, quantization could pull a kick and hi-hat struck a few ms apart
# exactly together and change the number of IOI events. Fixed in advance.
IOI_EVENT_TOLERANCE = 0.02

# A clip whose timing deviations have a standard deviation below this
# (seconds) has no timing structure to measure, e.g. a fully quantized
# clip. Its scale-free structure features are set to 0 by convention, in
# every condition, instead of being undefined.
MIN_TIMING_SPREAD = 1e-9

# --- Evaluation ----------------------------------------------------------

# A condition counts as "reliably detected" by a model when F1 for the
# edited class is at least this value for EVERY held-out drummer.
# Fixed before the final experiments were run.
RELIABLE_F1 = 0.9

HUMAN = "human"
