# How the code works

The pipeline has three steps, each a script in `src/`:

```
GMD MIDI ──build_dataset──▶ features.csv ──run_experiments──▶ results/experiments/
                                         └─run_ablation────▶ results/ablation/
```

Three lists define the experiment, each in exactly one place:

| What | Where | Used by |
|---|---|---|
| Conditions | `CONDITIONS` in `src/data/conditions.py` | build, experiments, plots, tests |
| Features and their groups | `FEATURE_GROUPS` in `src/features/__init__.py` | build, models, ablation |
| Models | `MODELS` in `src/models/classifiers.py` | experiments, ablation |

To add a condition, feature or model, add it to its list; everything else
picks it up.

## 1. Building the dataset (`src/build_dataset.py`)

For every GMD file:

1. **Scope filter** (`data/gmd.py`): keep ordinary 4/4 grooves. Fills, other
   meters, and styles labelled swing, shuffle or triplet are skipped, because
   a swung groove forced onto a straight grid would be detectable for reasons
   unrelated to quantization. GMD's style labels are used because onset
   positions alone did not separate swung from straight files. Plain `jazz`
   is kept (its label does not say whether it swings).
2. **Load** (`common/midi.py`): read the drum notes. `load_midi` also restores
   drum hits whose note-off has the same tick as the note-on, which
   pretty_midi would otherwise drop.
3. **Grids**: 1/16 and 1/32 grids are placed in MIDI ticks and converted to
   seconds through the file's tempo map, so they follow the recorded tempo.
4. **Clips** (`data/clips.py`): cut into consecutive, non-overlapping 2-bar
   clips from the first downbeat. A note belongs to the clip containing its
   nearest 1/16 grid point. The incomplete clip at the end is dropped.
5. **Conditions** (`data/conditions.py`): every condition is applied to the
   *whole* performance. The notes, their order and pitches never change, so
   each note keeps its original clip in every condition and every version of
   a clip contains the same notes.
6. **Features** (`features/`): measured on each clip's notes in each condition.

Output: one row per clip and condition in `data/features/features.csv`, a
`features_skipped.csv` log, and `results/dataset/DATASET.md` with counts and
mean feature values per condition (a quick check that every edit does what it
should).

### Conditions

| Family | Conditions | What it does | RQ |
|---|---|---|---|
| `human` | `human` | original | reference |
| `quantize` | `quantize_16_05` … `quantize_16_100` (5, 10, 15, 20, 25, 35, 50, 75, 100%), `quantize_32_100` | move each onset that share of the way to its nearest 1/16 (or 1/32) grid point | 1 |
| `velocity` | `velocity_25` … `velocity_100` | move each velocity that share of the way to 80; timing unchanged | 1 |
| `humanize` | `humanize_{uniform,gaussian,pink}_{5,10,20}ms`, `…_matched` | full 1/16 quantization plus timing jitter; velocity unchanged | 2, 3 |
| `shuffle` | `deviations_shuffled` | each clip's own human deviations, randomly reassigned among its notes | 3 |

Humanization details (`data/humanize.py`):

- The three jitter kinds have the **same SD at each level**, so they differ
  only in distribution shape (uniform vs Gaussian) and in correlation over
  time (pink = 1/f noise, slow drifts like human timing). Each note is
  jittered separately, as in DAW humanize functions, in onset order.
- Jitter is drawn once per performance and kind, and the 5, 10 and 20 ms
  levels use the same sequence at different scales (common random numbers),
  so level differences are not random noise.
- **Matched level**: within each clip the jitter is shifted and scaled to the
  mean and SD of that clip's human deviations. The deviation size is then the
  same as in the human clip, so only structure can tell them apart.
- **Shuffled control**: the clip keeps exactly its human set of deviations;
  only which note has which is random. This is the cleanest test of RQ3.
- Seeds come from the project seed plus the file and jitter kind
  (`rng_for`), so results do not depend on processing order.

### Features

All in seconds or MIDI velocity units; standard deviations use ddof=0.

| Group | Features | Meaning |
|---|---|---|
| context | `tempo`, `note_count` | the same in every version of a clip; let models interpret the rest |
| `mag_timing` | `mag_timing_abs_mean`, `mag_timing_std`, `mag_timing32_*`, `mag_{kick,snare,hihat}_timing_abs_mean` | size of deviations from the 1/16 and 1/32 grid, overall and per instrument |
| `mag_ioi` | `mag_ioi_mean`, `mag_ioi_std` | inter-onset intervals between events (onsets within 20 ms count as one event, so quantization cannot change the event count) |
| `mag_velocity` | `mag_velocity_mean`, `_std`, `_range` | size of velocity variation |
| `struct_timing` | `struct_timing_autocorr` | lag-1 autocorrelation of deviations over successive hits |
| | `struct_timing_position_r2` | share of deviation variance explained by position in the beat |
| | `struct_timing_instrument_r2` | share explained by the instrument |
| | `struct_chord_spread_ratio` | spread between *different* instruments on the same grid point, relative to the overall spread |
| `struct_velocity` | `struct_velocity_position_r2` | share of velocity variance explained by position in the beat (accents) |

Structure features are **scale-free**: multiplying all deviations by a
constant does not change them. Partial quantization does exactly that, so
structure features cannot detect it by construction. A clip with no timing
spread (fully quantized) gets 0 for its timing structure features in every
condition. Per-instrument timing, IOIs and chord spread can be undefined
(NaN), e.g. when a clip has no snare; models fill those with the training
median.

## 2. Experiments (`src/run_experiments.py`)

Each edited condition is its own balanced binary task: the human version of
every clip (class 0) against that condition's version of the same clips
(class 1). `models/evaluation.py`:

- **Validates** the table: every source clip has every condition once,
  versions agree on drummer, file, tempo and note count, and features are
  usable.
- **Leave-one-drummer-out**: train on all drummers but one, test on the
  held-out drummer, for every drummer. All versions of a clip and all clips
  of a performance stay with their drummer; the code checks for leakage.
- **Models** (`models/classifiers.py`) with fixed settings and no
  hyperparameter search: logistic regression, random forest, RBF SVM, and a
  timing rule (one cutoff on mean absolute timing deviation, direction learned
  from training data) as a reference. Missing-value filling and scaling are
  fitted on the training fold only.
- **Metrics**: accuracy, precision, recall and F1 for the edited class, per
  held-out drummer and pooled.
- **Reliability**: a condition is reliably detected by a model when F1 ≥ 0.9
  for every held-out drummer (`RELIABLE_F1` in `config.py`, fixed before the
  final run). `reliability.csv` gives, per model and series, the lowest
  quantization strength from which detection stays reliable, and the largest
  jitter up to which it stays reliable.

Outputs in `results/experiments/`: `RESULTS.md`, `metrics.csv`,
`fold_metrics.csv`, `reliability.csv`, `coefficients.csv` (logistic
regression), `run_config.json` (settings, input hash, versions) and figures.
Full predictions go to `data/features/predictions.csv`.

## 3. Feature ablation (`src/run_ablation.py`)

Reruns the evaluation (logistic regression and random forest) with:
all features, magnitude only, structure only, and all features minus one
group. Outputs in `results/ablation/`: `ABLATION.md`, `ablation_metrics.csv`,
figures and logistic-regression coefficients.

## Tests (`tests/`)

Synthetic MIDI files (`tests/synthetic.py`) and small arrays; no GMD needed.
They check the grid and clip logic, every condition's guarantees (nominal
jitter size, matched mean/SD per clip, shuffle keeps the deviation set,
determinism, structure features unchanged under partial edits), the feature
definitions, the timing rule, leakage-free folds and training-only scaling.

## Things to keep in mind when interpreting results

- Full quantization and full velocity normalization are detectable by
  construction (deviation or velocity spread exactly 0): sanity checks.
- Matched jitter has the human mean and SD, but its mean *absolute* deviation
  still depends on the distribution shape; the shuffled control is exact.
- Pink jitter is scaled over the whole performance, so within a 2-bar clip
  its SD is smaller than the nominal level.
- Partial velocity normalization gives non-integer velocities; features use
  them directly, exported MIDI rounds them.
- Results describe detection of synthetic edits of this GMD subset, not what
  listeners hear.
