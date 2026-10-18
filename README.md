# Groove Detection

Can a classifier tell real drum performances from quantized and artificially
humanized versions? Data: Groove MIDI Dataset (GMD).

Current stage: **first classifier results** — logistic regression and a timing
rule evaluated on matched conditions with each drummer held out in turn.
See [CODE_EXPLAINER.md](CODE_EXPLAINER.md) for the pipeline and table columns.
The checked dataset summary and plots are in [results/DATASET_CHECKS.md](results/DATASET_CHECKS.md).
Classifier metrics and plots are in [results/baseline/MODEL_RESULTS.md](results/baseline/MODEL_RESULTS.md).

## Layout

```
src/
  build_dataset.py       entry point: GMD -> human / matched condition features
  train_baseline.py      entry point: features -> held-out-drummer evaluation
  models/
    baseline.py          logistic regression, timing rule, grouped evaluation
  common/
    config.py            clip length, seed, data paths
    midi.py              drum notes, tempo-aware 1/16 and 1/32 grids, bar lines
  data/
    gmd.py               info.csv loading / filtering
    clips.py             cut performances into 2-bar clips
    quantize.py          partial timing quantization and velocity normalization
  features/
    magnitude.py         timing / IOI / velocity features
tests/                   synthetic data, no GMD needed
results/                 tracked summaries and plots for sharing
  baseline/              metrics, fold results, configuration, plots, report
data/                    NOT in git
  groove/                GMD as downloaded (contains info.csv)
  generated/             optional transformed full-performance MIDI exports
  features/              generated feature tables
```

## Setup

```bash
python -m venv env
env/bin/pip install -r requirements.txt
```

Download GMD into `data/groove/` so that `data/groove/info.csv` exists.

## Running

Run every command from the repository root.

```bash
env/bin/python -m src.build_dataset
```

Build all seven conditions (including the same human rows):

```bash
env/bin/python -m src.build_dataset --conditions
```

Optionally also export transformed full performances for playback:

```bash
env/bin/python -m src.build_dataset --conditions --midi-output data/generated
```

Run the six balanced human-versus-edited experiments:

```bash
env/bin/python -m src.train_baseline
```

Small summaries go into `results/baseline/`. Full held-out predictions go into
ignored `data/features/baseline_predictions.csv`.

```bash
env/bin/python -m pytest
```
