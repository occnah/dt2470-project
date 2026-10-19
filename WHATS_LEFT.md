# What's left

Updated: 19 October 2026.

The data pipeline, first classifier evaluation, and the feature-ablation check
are complete. Remaining work focuses on the model comparison, interpretation,
and writing the report.

## Completed

- [x] Prepare two-bar clips from GMD, restricted to ordinary 4/4 beats/grooves
  (fills, non-4/4, and swing/triplet-feel performances excluded — see
  CODE_EXPLAINER.md): 453 of 1,150 performances, 9,894 clips. This drops
  drummer10 entirely (their only session is all jazz-swing).
- [x] Fix IOI features to group onsets within 20ms into one event, applied to
  every condition, so quantization can't silently change the IOI event count.
- [x] Generate matched quantization and velocity-normalization conditions on
  the filtered clips: 69,258 feature rows across seven conditions.
- [x] Extract timing, inter-onset interval, and velocity features.
- [x] Evaluate logistic regression and a training-only timing rule using
  leave-one-drummer-out evaluation across the nine remaining drummers.
  Re-running after the filter above changed scores negligibly (within
  ~0.01 F1 at every quantization level) — the dataset fix did not change the
  earlier conclusions, it just removed a confound from how they were reached.
- [x] Save metrics, confusion matrices, per-drummer results, and plots.
- [x] **Feature ablation** (was the top remaining priority): reran the four
  1/16 experiments without `timing_32_abs_mean`/`timing_32_std`
  (`src/ablation.py`, results in `results/ablation/ABLATION_RESULTS.md`).
  The original concern — a 50%+ shortcut from the 1/16/1/32 relationship —
  does **not** hold: removing the 1/32 features costs almost nothing at
  75%/100% (where that relationship is strongest), but costs ~9-10 F1 points
  at 25%/50% (where it's weakest). The 1/32 features carry real signal where
  detection is hardest, not an inflated shortcut where it was already easy.
- [x] Update the code explainer and verify all 74 tests pass.

## Remaining, in recommended order

### 1. Complete the model comparison

- [ ] Add Random Forest and SVM using the same matched examples, numerical
  feature whitelist, and held-out drummers.
- [ ] Keep settings fixed, or perform tuning using grouped validation entirely
  inside each training fold.
- [ ] Update the comparison tables and figures for all models.

### 2. Interpret the findings

- [ ] Inspect representative false positives and false negatives, including
  differences between drummers, styles, and tempos.
- [ ] State an operational definition of “reliably detectable”; clearly label
  any criterion chosen after viewing results as exploratory.
- [ ] Explain limitations: synthetic editing fingerprints, two-bar clip
  exclusions, uneven drummer coverage (including drummer10's exclusion above),
  and the absence of a listening study.

### 3. Write and finalize the report

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
