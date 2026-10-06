# Groove Detection

Can a classifier tell real drum performances from quantized and artificially
humanized versions? Data: Groove MIDI Dataset (GMD).

Current stage: the **human baseline** — features for 2-bar clips of the
original, unmodified performances.

## Layout

```
src/
  build_dataset.py       entry point: GMD -> data/features/human_features.csv
  common/
    config.py            clip length, seed, data paths
    midi.py              drum notes, tempo-aware 1/16 grid, bar lines
  data/
    gmd.py               info.csv loading / filtering
    clips.py             cut performances into 2-bar clips
    quantize.py          quantization (not used yet)
  features/
    magnitude.py         timing / IOI / velocity features
tests/                   synthetic data, no GMD needed
data/                    NOT in git
  raw/                   GMD as downloaded (contains info.csv)
  features/              generated feature tables
```

## Setup

```bash
python -m venv env
env/bin/pip install -r requirements.txt
```

Download GMD into `data/raw/` so that `data/raw/info.csv` exists.

## Running

Run every command from the repository root.

```bash
env/bin/python -m src.build_dataset --input data/raw --output data/features/human_features.csv
```

```bash
env/bin/python -m pytest
```
