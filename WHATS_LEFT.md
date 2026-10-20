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
- [x] Generated reports and figures in `results/`; tests for every step.

## Remaining

- [ ] Inspect representative errors (false positives/negatives) by drummer,
  style and tempo, if the discussion needs it.
- [ ] Write the report from `results/`.
