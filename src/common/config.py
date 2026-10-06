"""Project-wide constants."""

from pathlib import Path

SEED = 42
BARS_PER_CLIP = 2

# The Groove MIDI Dataset as downloaded (contains info.csv).
RAW_DIR = Path("data/raw")

# Feature tables.
FEATURES_DIR = Path("data/features")

HUMAN = "human"
