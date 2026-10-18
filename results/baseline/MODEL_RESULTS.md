# Baseline classifier results

Run: 2026-10-18T22:00:43+02:00.

## Method

Six separate balanced binary experiments compare original clips with one edited condition at a time. Edited clips are the positive class (1); human clips are class 0.

Each experiment has 21,608 clips. Leave-one-drummer-out evaluation trains on 9 drummers and tests on the remaining drummer, repeating for all 10 drummers. Every row is tested exactly once per experiment and method. All versions of a clip, and all clips from a performance, stay with their drummer.

- Logistic regression uses the 10 existing numerical features, StandardScaler fitted only on the training fold, and fixed settings: C=1, lbfgs, max_iter=2000, seed=42. The probability decision threshold is 0.5. No hyperparameter search or test-driven feature selection was performed.
- The timing rule predicts an edit when mean absolute grid deviation is at or below a cutoff chosen only from training rows. The cutoff maximizes balanced accuracy; tied objectives choose the lowest cutoff. It uses the 1/16 deviation for the four 1/16 experiments and the velocity-only control, and the 1/32 deviation for the 1/32 experiment.
- Both methods receive only rhythmic features. IDs, drummer labels, original GMD split labels, condition names, and transformation settings are excluded from classifier inputs.
- These drummer-held-out folds replace the supplied GMD split for this experiment. There is no additional untouched test cohort; these results evaluate the fixed baseline protocol.

## Equal-weight scores across drummers

Each drummer contributes equally below. Values are mean ± sample standard deviation across held-out drummers; this spread is not a confidence interval. Precision, recall, and F1 refer to edited clips. Precision and F1 are recorded as 0 when no positive clips are predicted.

| Condition | Method | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1/16 at 25% | Logistic regression | 0.729 ± 0.065 | 0.715 ± 0.091 | 0.794 ± 0.102 | 0.745 ± 0.058 |
| 1/16 at 25% | Timing threshold | 0.635 ± 0.019 | 0.629 ± 0.037 | 0.684 ± 0.125 | 0.647 ± 0.044 |
| 1/16 at 50% | Logistic regression | 0.943 ± 0.051 | 0.910 ± 0.080 | 0.991 ± 0.013 | 0.947 ± 0.045 |
| 1/16 at 50% | Timing threshold | 0.801 ± 0.047 | 0.809 ± 0.064 | 0.802 ± 0.105 | 0.799 ± 0.053 |
| 1/16 at 75% | Logistic regression | 0.971 ± 0.019 | 0.973 ± 0.030 | 0.971 ± 0.025 | 0.971 ± 0.019 |
| 1/16 at 75% | Timing threshold | 0.954 ± 0.021 | 0.959 ± 0.039 | 0.952 ± 0.047 | 0.954 ± 0.022 |
| 1/16 at 100% | Logistic regression | 1.000 ± 0.001 | 0.999 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.001 |
| 1/16 at 100% | Timing threshold | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 1/32 at 100% | Logistic regression | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.001 |
| 1/32 at 100% | Timing threshold | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Velocity only | Logistic regression | 0.998 ± 0.003 | 0.997 ± 0.006 | 1.000 ± 0.001 | 0.998 ± 0.003 |
| Velocity only | Timing threshold | 0.500 ± 0.000 | 0.050 ± 0.158 | 0.000 ± 0.001 | 0.001 ± 0.002 |

## Pooled scores

These metrics pool all held-out predictions. Drummers with more clips have more influence here; compare with the equal-weight scores above.

| Condition | Method | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/16 at 25% | Logistic regression | 0.736 | 0.691 | 0.853 | 0.764 | 6,688 | 4,116 | 1,585 | 9,219 |
| 1/16 at 25% | Timing threshold | 0.638 | 0.615 | 0.737 | 0.671 | 5,820 | 4,984 | 2,838 | 7,966 |
| 1/16 at 50% | Logistic regression | 0.933 | 0.887 | 0.994 | 0.937 | 9,432 | 1,372 | 70 | 10,734 |
| 1/16 at 50% | Timing threshold | 0.810 | 0.792 | 0.840 | 0.815 | 8,417 | 2,387 | 1,726 | 9,078 |
| 1/16 at 75% | Logistic regression | 0.973 | 0.965 | 0.982 | 0.974 | 10,425 | 379 | 198 | 10,606 |
| 1/16 at 75% | Timing threshold | 0.955 | 0.949 | 0.963 | 0.956 | 10,242 | 562 | 405 | 10,399 |
| 1/16 at 100% | Logistic regression | 1.000 | 1.000 | 1.000 | 1.000 | 10,802 | 2 | 0 | 10,804 |
| 1/16 at 100% | Timing threshold | 1.000 | 1.000 | 1.000 | 1.000 | 10,804 | 0 | 0 | 10,804 |
| 1/32 at 100% | Logistic regression | 1.000 | 1.000 | 1.000 | 1.000 | 10,803 | 1 | 0 | 10,804 |
| 1/32 at 100% | Timing threshold | 1.000 | 1.000 | 1.000 | 1.000 | 10,804 | 0 | 0 | 10,804 |
| Velocity only | Logistic regression | 0.999 | 0.997 | 1.000 | 0.999 | 10,775 | 29 | 1 | 10,803 |
| Velocity only | Timing threshold | 0.500 | 0.500 | 0.000 | 0.000 | 10,803 | 1 | 10,803 | 1 |

## Figures

![Detection versus 1/16 quantization strength](f1_by_strength.png)

The 1/32 grid and velocity-only condition are separate experiments and are not points on the 1/16 strength curve.

![F1 for each held-out drummer](f1_by_drummer.png)

![Confusion matrices](confusion_matrices.png)

## Checks and interpretation

- All source clips have exactly one row for each of the seven conditions. Features are finite and matching versions have consistent drummer, source file, and note count.
- Training and test drummers, source performances, and source clip IDs are disjoint in every fold. Stored held-out predictions have exactly one row per clip, method, and experiment.
- The velocity-only timing control must have 50% accuracy because each human/edited pair has identical timing. Its F1 need not be 0.5: F1 depends on which class receives the predictions, and a tied training cutoff can yield no positive predictions.
- Full grid snapping and constant velocity leave direct signatures in the selected features. At 50% or stronger movement to the 1/16 grid, a geometric relation between the 1/16 and 1/32 deviation features can also help the classifier. High scores measure detection of these synthetic edits, not general authorship or listener judgments.
- Results apply to the retained two-bar GMD subset, including straight, swung, and triplet patterns. The original short-clip exclusions and uneven drummer coverage still apply.
- No formal numerical reliability criterion was fixed for this run. Scores and their drummer-to-drummer variation are reported without declaring a universal detectability threshold.

## Reproduce and inspect

From the project root:

```bash
env/bin/python -m src.train_baseline
```

- `metrics.csv`: pooled metrics plus equal-weight drummer means and SDs.
- `fold_metrics.csv`: each held-out drummer, training drummer IDs, learned timing cutoff, and confusion counts.
- `run_config.json`: fixed settings, input-table SHA-256, feature list, and package versions.
- `data/features/baseline_predictions.csv` in the project: full held-out labels/predictions, kept outside Git with other generated data when using the default path.

Input SHA-256: `e91c459890fbb35de293f26815f2aab7643c1c9202ba9e69405a668c0dfaac02`.
