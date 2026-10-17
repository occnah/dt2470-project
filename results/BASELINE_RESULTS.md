# Baseline classification results - 17 October 2026

First classifier run, after two methodology fixes to the dataset (see
`CODE_EXPLAINER.md`): restricting to ordinary 4/4 beats/grooves (excluding
fills and swing/triplet-feel styles), and grouping onsets within 20ms into
one event for IOI features. Everything below supersedes the counts in
`results/DATASET_CHECKS.md`, which predate both fixes.

## Dataset after filtering

- Input: 1,150 GMD MIDI performances.
- **Used: 453 performances** (down from 577 pre-filter).
- Skipped: 697 performances.
  - `not a beat (fill)`: 647
  - `shorter than one clip`: 27
  - `not 4/4`: 12
  - `swing/triplet style`: 11
- Human baseline: **9,894 two-bar clips** (down from 10,804).
- Conditions table: **69,258 rows** (9,894 x 7 conditions).

Most of the old "shorter than one clip" skips (573) were actually fills,
which are inherently short; filtering fills out directly dropped file count
more than clip count, since the beats kept are typically longer than the
fills excluded.

**Caveat on drummer coverage**: drummer10's only session (`jazz-swing`,
all performances) is entirely excluded by the swing filter. The
classification splits below therefore only ever train/test on 9 of the
10 GMD drummers, not 10. This should be noted wherever drummer coverage is
discussed.

## Classification setup

`src/train_baseline.py`: drummer-disjoint train/test split
(`GroupShuffleSplit`, 25% of drummers held out, seed 42), logistic
regression and random forest on the 10 magnitude features
(`note_count`, `timing_abs_mean`, `timing_std`, `ioi_mean`, `ioi_std`,
`velocity_mean`, `velocity_std`, `velocity_range`, `timing_32_abs_mean`,
`timing_32_std`). No identifiers, condition metadata, or tempo in the
inputs.

- Train drummers: drummer1, drummer3, drummer4, drummer5, drummer7, drummer9 (12,816 rows)
- Test drummers: drummer2, drummer6, drummer8 (6,972 rows)

## Results: human vs. each condition

| Condition | Logistic regression | Random forest | Top features |
| --- | ---: | ---: | --- |
| `quantized_16_25` | 0.688 | 0.702 | timing_abs_mean, timing_std, timing_32_* |
| `quantized_16_50` | 0.896 | 0.901 | timing_abs_mean, timing_std |
| `quantized_16_75` | 0.962 | 0.968 | timing_std, timing_abs_mean |
| `quantized_16_100` | 1.000 | 1.000 | timing_32_std, timing_32_abs_mean |
| `quantized_32_100` | 1.000 | 1.000 | timing_32_std, timing_32_abs_mean |
| `velocity_normalized` | 0.999 | 0.999 | velocity_std, velocity_range |

(Accuracy on the held-out drummers' clips.)

## Interpretation

- **The real finding is the 25% -> 50% -> 75% gradient**: detection accuracy
  rises smoothly with quantization strength, and in every case timing
  features (not velocity or note count) drive the decision. This is
  consistent with the classifier genuinely picking up on timing deviation
  rather than a confound.
- **`quantized_16_100` and `quantized_32_100` reaching exactly 1.000 is
  expected, not a result worth reporting on its own**: full quantization
  forces `timing_abs_mean`/`timing_std` (or the 1/32 equivalents) to
  exactly zero for every clip, which real human playing essentially never
  does. This is trivially separable by construction.
- **`velocity_normalized` at 0.999 is the same kind of trivial-by-construction
  case, not a validated "negative control"**: setting every drum hit to a
  constant velocity collapses `velocity_std`/`velocity_range` to exactly
  zero, which is just as easy to detect as timing collapsing to zero.
  The useful signal here is that timing-feature importances are ~0 for this
  condition (the model correctly isn't using timing to detect a
  velocity-only change), which is a basic correctness check on the
  features, not evidence that detection is hard in general.
- No model selection or hyperparameter tuning has been done; these are
  default-parameter baselines meant to establish whether the signal exists
  at all, which it clearly does for partial quantization.

## Outputs in the project

- `results/BASELINE_RESULTS.md` (this file)
- `src/train_baseline.py`
- `data/features/conditions_features.csv`, `data/features/human_features.csv` (not tracked in git)

Run environment: same as `results/DATASET_CHECKS.md` (Python 3.12.14,
NumPy, pandas, mido, scikit-learn).
