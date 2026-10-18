# What's left

Updated: 18 October 2026.

The data pipeline and first classifier evaluation are complete. Remaining work
focuses on validating the interpretation, comparing models, and writing the report.

## Completed

- [x] Prepare 10,804 original two-bar clips from GMD.
- [x] Generate matched quantization and velocity-normalization conditions:
  75,628 feature rows across seven conditions.
- [x] Extract timing, inter-onset interval, and velocity features.
- [x] Evaluate logistic regression and a training-only timing rule using
  leave-one-drummer-out evaluation across ten drummers.
- [x] Save metrics, confusion matrices, per-drummer results, and plots.
- [x] Update the code explainer and verify all 64 tests pass.

## Remaining, in recommended order

### 1. Feature ablation — highest priority

- [ ] Repeat the four 1/16 quantization experiments without the 1/32 timing
  features, keeping the same folds and model settings.
- [ ] Compare full-feature and ablation scores, including variation between
  drummers, to establish how much detection depends on the relationship between
  the two grids.

At 50% or stronger 1/16 quantization, the mean absolute deviations from the two
grids become equal by construction. This provides a potential shortcut for the
classifier and affects how we interpret its strong performance.

### 2. Complete the model comparison

- [ ] Add Random Forest and SVM using the same matched examples, numerical
  feature whitelist, and held-out drummers.
- [ ] Keep settings fixed, or perform tuning using grouped validation entirely
  inside each training fold.
- [ ] Update the comparison tables and figures for all models.

### 3. Interpret the findings

- [ ] Inspect representative false positives and false negatives, including
  differences between drummers, styles, and tempos.
- [ ] State an operational definition of “reliably detectable”; clearly label
  any criterion chosen after viewing results as exploratory.
- [ ] Explain limitations: synthetic editing fingerprints, two-bar clip
  exclusions, uneven drummer coverage, and the absence of a listening study.

### 4. Write and finalize the report

- [ ] Write the research questions, dataset and preprocessing, quantization
  conditions, features, models, evaluation, results, and discussion.
- [ ] Include the final comparison figures, uncertainty across drummers, and
  the required dataset/method references.
- [ ] Answer how detection changes with quantization strength, with conclusions
  limited to the tested conditions and dataset.
- [ ] Check that the report, code explainer, results, and reproduction commands
  agree after the final experiments.

## Optional future work

Artificial humanization, structural timing features such as autocorrelation and
coordination between instruments, and a listening study can be discussed as
extensions to the core quantization-detection project.
