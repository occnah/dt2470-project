# Groove detection pipeline

## Run it

From the project root, with the dependencies installed in `env/`:

```bash
env/bin/python -m src.build_dataset
env/bin/python -m src.build_dataset --conditions
env/bin/python -m src.train_baseline
env/bin/python -m pytest -q
```

The input now defaults to the existing `data/groove/` folder. The commands write
`data/features/human_features.csv` and `data/features/conditions_features.csv`.
Each also writes a `_skipped.csv` listing unusable source files and reasons.
Use `--input` and `--output` to override the locations.

## What the code does

- `common/midi.py`: builds 1/16 and 1/32 grids in MIDI ticks, then converts them
  through the performance's tempo map. Both quantization and timing features use
  these grids. Exact halfway ties use the earlier grid point. The loader also
  restores raw drum note-ons omitted by the MIDI duration parser, with their
  original onset/velocity and zero duration, so every recorded drum hit is counted.
- `build_dataset.py` main loop: before processing, each file is kept only if it
  is an ordinary 4/4 beat/groove — `beat_type == "beat"` (fills excluded),
  `time_signature == "4-4"`, and `style` not marked `swing` or `triplet`. This
  filter only runs when GMD's `info.csv` was loaded; it has no effect on
  `process_file` itself, which the synthetic tests call directly. The goal is
  to measure "human vs quantized" rather than "swing vs straight" — a swung
  groove forced onto a straight grid would be detectable for reasons
  unrelated to quantization.
- `data/quantize.py`: moves each onset toward its nearest grid point:
  `new = original + strength * (target - original)`. It also provides a separate
  transformation setting every drum velocity to 80. MIDI helpers return copies,
  preserve pitches and note durations, and leave non-drum notes unchanged.
- `build_dataset.py`: cuts the original into non-overlapping two-bar clips once.
  Every version uses that original note membership, even if a transformed onset
  moves across the clip boundary. Empty clips and unusable files are counted.
- `features/magnitude.py`: extracts timing, inter-onset interval (IOI), and velocity
  summaries from those matched notes. Existing feature names are preserved;
  `timing_32_abs_mean` and `timing_32_std` add the finer-grid measurements.
  IOIs are computed between onset *events* (`group_onset_events`), not raw
  distinct onset times: onsets within `IOI_EVENT_TOLERANCE` (20ms) of each
  other — e.g. a kick and hi-hat struck together — merge into one event.
  Applied identically to every condition, so quantization pulling two such
  onsets exactly together can't change the IOI event count on its own and
  leak detectability through a counting artifact rather than real timing
  deviation.
- `tests/test_quantize.py`: checks partial strengths, tempo changes, boundary
  membership, independent velocity normalization, and optional MIDI exports.
- `models/baseline.py`: validates matched clips, trains logistic regression and
  a timing-threshold rule, and evaluates each on held-out drummers.
- `train_baseline.py`: runs the six comparisons and saves metrics, plots, run
  settings, and full held-out predictions. `tests/test_baseline.py` checks the
  pairing, thresholds, feature whitelist, and training-only scaling.
- `ablation.py`: reruns the four 1/16 experiments with `timing_32_abs_mean`/
  `timing_32_std` excluded from logistic regression's inputs, to check
  whether their relationship to the 1/16 deviation at 50%+ quantization
  strength was giving it a shortcut (see `WHATS_LEFT.md`). Reuses
  `evaluate_table`'s new `conditions`/`feature_columns` parameters rather
  than duplicating the leave-one-drummer-out protocol.

## Conditions and identifiers

Each usable source clip has seven rows in the conditions table:

| `condition` | Timing | Velocity |
| --- | --- | --- |
| `human` | Original | Original |
| `quantized_16_25` | 25% toward 1/16 grid | Original |
| `quantized_16_50` | 50% toward 1/16 grid | Original |
| `quantized_16_75` | 75% toward 1/16 grid | Original |
| `quantized_16_100` | Fully on 1/16 grid | Original |
| `quantized_32_100` | Fully on 1/32 grid | Original |
| `velocity_normalized` | Original | Constant 80 |

`clip_id` uniquely identifies a row. `source_clip_id` links its seven versions
and serves as the stable source identifier for paired/provenance analysis.
`source_file`, `drummer`, `style`, `beat_type`, and `time_signature` retain
the original GMD metadata (the latter two are also the fields the main-scope
filter above is applied to). `split` is the existing GMD split, not a new
drummer-disjoint split. `subdivision`, `strength`,
and `normalized_velocity` describe the transformation; inapplicable values are
blank. Human and velocity-only rows have strength 0.

Times and timing features are in seconds. `tempo` is quarter notes per minute.
`timing_abs_mean` and `timing_std` measure deviations from the 1/16 grid;
`timing_32_*` uses the 1/32 grid. IOIs are between distinct onset times, pooling
all drums. Exact simultaneous hits count as one onset event; a clip with only
one distinct onset gets missing IOI values. Velocity summaries pool all drum
notes and are in MIDI velocity units.

## Optional MIDI exports and scope

Add `--midi-output data/generated` to the conditions command to save six
transformed full performances per usable file, under condition/source paths.
These are full performances, not separately cut clip files. Original MIDI is
never rewritten. MIDI exports round fractional ticks to the native resolution;
the feature table is computed directly in memory before that rounding. Use
`load_midi` when reading files if zero-duration drum hits must be retained.

Dataset construction preserves the original GMD split labels. The classifier
uses the separate drummer-held-out evaluation below. No style filtering or
artificial humanization is applied.

## Full run on 11 October 2026

577 of 1,150 performances produced 10,804 original clips; 573 performances were
too short for a complete two-bar clip under the existing clipping rule. The
conditions table contains 75,628 rows, and 3,462 transformed MIDI files were
exported. The loader recovered 796 omitted drum hits across 204 usable files.
The 36 synthetic tests and full-data checks pass. The audit and feature plots are
saved in [results/DATASET_CHECKS.md](results/DATASET_CHECKS.md), with the plot in
`results/dataset_checks.png`, after checking every table row and MIDI export.
These small summaries are tracked in Git; full feature tables and generated
MIDI remain ignored under `data/`. Classifier results are reported separately.

## Baseline evaluation on 18 October 2026

Each experiment compares human clips with one of the six edited conditions,
with edited clips as class 1. It trains on nine drummers and tests on the tenth,
rotating through all ten. Related clips and versions stay together. The original
GMD split is not used for this evaluation; there is no extra untouched test set.

Logistic regression uses the ten listed numerical features through a
`StandardScaler` fitted inside each training fold. Settings are fixed: C=1,
lbfgs, max_iter=2000, seed=42, probability cutoff=0.5. IDs, drummer, condition,
and transformation metadata never enter the classifier.

The comparison rule predicts an edit for a small mean absolute timing deviation.
Its cutoff maximizes balanced accuracy on training rows only, with tied cutoffs
resolved toward the lowest value. It uses the matching 1/16 or 1/32 grid;
velocity-only uses 1/16 timing as a control and gets exactly 50% accuracy.

The full run produced 120 method/fold rows and 259,296 held-out predictions;
all 64 tests pass. Mean F1 across drummers for logistic regression is 0.745 at
25% quantization and 0.947 at 50%, versus 0.647 and 0.799 for the timing rule.
Full quantization and velocity normalization are almost perfectly classified.
These scores concern synthetic edits of this dataset; direct feature signatures
can make detection easy and do not establish what listeners can hear.

See [results/baseline/MODEL_RESULTS.md](results/baseline/MODEL_RESULTS.md) for all
conditions, confusion matrices, and per-drummer plots. `metrics.csv` contains
pooled scores plus equal-weight drummer means and sample SDs; `fold_metrics.csv`
records each test drummer and training cutoff. `run_config.json` stores the
fixed settings, feature list, input hash, and package versions. Full predictions
are in ignored `data/features/baseline_predictions.csv`.

Note: this run predates the 4/4-beat/swing-triplet filter and the IOI event
grouping fix above. Re-running `train_baseline` on the filtered dataset gave
nearly identical scores (within ~0.01 F1 at every quantization level), so the
filter did not change the conclusions here — see the regenerated
`results/baseline/` files for the current numbers; the counts/scores in this
section are left as originally written and are now slightly stale.

## Feature ablation

`src/ablation.py` reruns the four 1/16 experiments with the 1/32 timing
features removed from logistic regression, since at 50%+ 1/16 quantization
strength the 1/16 and 1/32 deviation features become related by
construction (moving further toward the 1/16 grid also reduces the 1/32
deviation). The timing-threshold baseline is unaffected by design on these
conditions — it already uses `timing_abs_mean` alone — and is checked, not
assumed, to be identical in both runs. See
[results/ablation/ABLATION_RESULTS.md](results/ablation/ABLATION_RESULTS.md)
once it's been run (`env/bin/python -m src.ablation`) for the comparison.
