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
