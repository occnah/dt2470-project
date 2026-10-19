"""Project-wide constants."""

from pathlib import Path

SEED = 42
BARS_PER_CLIP = 2

# The Groove MIDI Dataset as downloaded (contains info.csv).
RAW_DIR = Path("data/groove")

# Feature tables.
FEATURES_DIR = Path("data/features")

HUMAN = "human"

# (condition name, note subdivision, movement toward the nearest grid point).
QUANTIZATION_CONDITIONS = (
    ("quantized_16_25", 16, 0.25),
    ("quantized_16_50", 16, 0.50),
    ("quantized_16_75", 16, 0.75),
    ("quantized_16_100", 16, 1.0),
    ("quantized_32_100", 32, 1.0),
)
NORMALIZED_VELOCITY = 80

# Main experiment scope: ordinary 4/4 beats/grooves only. Fills and
# performances explicitly marked as a swing or triplet feel are excluded
# so the task measures "human vs quantized" rather than "swing vs
# straight" (a swung groove forced onto a straight 1/16 grid would be
# trivially easy to detect for reasons unrelated to quantization).
VALID_BEAT_TYPE = "beat"
VALID_TIME_SIGNATURE = "4-4"
EXCLUDED_STYLE_KEYWORDS = ("swing", "triplet")

# Onsets within this many seconds of each other count as one event for
# inter-onset-interval (IOI) purposes, applied identically to every
# condition. Without this, quantization could move two notes that were a
# few milliseconds apart (e.g. kick + hi-hat) exactly on top of each
# other, changing the number of distinct onset events and letting IOI
# features partly detect quantization through that artifact rather than
# through genuine timing deviation. Chosen once, in advance of looking at
# any classifier results, and not tuned afterward.
IOI_EVENT_TOLERANCE = 0.02
