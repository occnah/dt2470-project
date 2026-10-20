# What's left

Updated: 20 October 2026.

## Code: complete

- [x] Dataset: 4/4 grooves, fills and swing/shuffle/triplet styles excluded,
  2-bar clips, tempo-aware grids.
- [x] Conditions for all three research questions: 1/16 quantization at nine
  strengths, 1/32, velocity normalization at four strengths, uniform /
  Gaussian / 1/f humanization at 5, 10, 20 ms and human-matched size, and the
  shuffled-deviation control.
- [x] Magnitude (`mag_*`) and structure (`struct_*`) features.
- [x] Logistic regression, random forest, SVM and a timing rule, evaluated
  leave-one-drummer-out with a reliability criterion fixed in advance.
- [x] Feature-group ablation and logistic-regression coefficients (RQ3).
- [x] Qualitative error analysis by drummer, style and tempo
  (`src/error_analysis.py`, `results/error_analysis/ERROR_ANALYSIS.md`):
  which drummer is hardest to detect depends on the edit type -- `drummer6`
  is specifically hard for quantization (false-negative rate 0.347 at 25%
  strength vs. 0.157 overall), `drummer2` specifically hard for velocity
  normalization, humanization and the shuffle control (0.20-0.39 vs.
  0.135-0.179 overall), consistent with the RQ3 finding that those edit
  types are detected through different feature groups. A handful of GMD
  sub-styles without a literal swing/shuffle/triplet label but a plausibly
  non-straight feel (`jazz/fast`, `jazz/fusion`, `latin/venezuelan-joropo`)
  recur as the worst style group across multiple conditions -- a residual
  gap in the keyword-based scope filter, not a classifier problem.
- [x] Generated reports and figures in `results/`; tests for every step.

## Remaining

- [ ] Finalize the report (first draft in Overleaf covers all three RQs and
  the error analysis above; needs a read-through from Karl and Ossian,
  especially the Introduction/Previous work, which neither has seen yet).
