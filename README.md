# Groove Detection

Can a classifier tell real drum performances from quantized and artificially
humanized versions, and at what level does each become reliably detectable?
Data: the Groove MIDI Dataset (GMD).

Research questions:

1. How much quantization (strength, grid size, velocity normalization) is needed
   before a classifier reliably detects it?
2. Can a classifier tell real drumming from quantized-then-humanized drumming,
   as produced by DAW-style "humanize" functions?
3. Which features carry the signal: the size of timing and velocity deviations,
   or their structure (correlation over time, metric position, coordination
   between instruments)?

## Layout

```
src/
  build_dataset.py      1. GMD -> data/features/features.csv (all conditions)
  run_experiments.py    2. features -> results/experiments/  (RQ1, RQ2)
  run_ablation.py       3. features -> results/ablation/     (RQ3)
  error_analysis.py     4. features + predictions -> results/error_analysis/
  common/
    config.py           paths, seed, dataset scope, analysis settings
    midi.py             MIDI loading, tempo-aware grids, bar lines, drum classes
  data/
    gmd.py              info.csv metadata and the scope filter
    clips.py            cut performances into 2-bar clips
    quantize.py         timing quantization and velocity normalization
    humanize.py         uniform / Gaussian / 1/f timing jitter
    conditions.py       THE list of conditions and how each is applied
  features/
    __init__.py         feature groups and the extraction entry point
    magnitude.py        mag_*    size of deviations
    structure.py        struct_* arrangement of deviations (scale-free)
  models/
    classifiers.py      THE list of models
    evaluation.py       validation, matched pairs, leave-one-drummer-out
  reporting/            markdown tables and figures for results/
tests/                  synthetic data, no GMD needed
results/                generated summaries and figures (tracked in git)
  dataset/              what the feature table contains
  experiments/          detection per condition and model
  ablation/             detection per feature group
  error_analysis/       false-positive/negative rates by drummer, style, tempo
data/                   NOT in git
  groove/               GMD as downloaded (contains info.csv)
  features/             feature table and full predictions
```

## Setup

```bash
python -m venv env
env/bin/pip install -r requirements.txt
```

Download GMD into `data/groove/` so that `data/groove/info.csv` exists.

## Running

Run every command from the repository root, in this order.

```bash
env/bin/python -m src.build_dataset
```

```bash
env/bin/python -m src.run_experiments
```

```bash
env/bin/python -m src.run_ablation
```

```bash
env/bin/python -m src.error_analysis
```

```bash
env/bin/python -m pytest
```

On a 10-core laptop, building takes about 3 minutes, the experiments about 35
minutes and the ablation about 50. Both experiment scripts accept `--models`
and `--conditions` to run a subset, and `--report-only` to redraw the report
and figures from saved results without retraining. Add
`--midi-output data/generated` to the build to save every edited performance
as MIDI for listening.
