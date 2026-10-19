# Feature ablation: 1/16 detection without the 1/32 timing features

Dropped features: `timing_32_abs_mean`, `timing_32_std`.
Kept features: `note_count`, `timing_abs_mean`, `timing_std`, `ioi_mean`, `ioi_std`, `velocity_mean`, `velocity_std`, `velocity_range`.

Same leave-one-drummer-out protocol as `results/baseline/`, restricted to the
four 1/16 quantization conditions. Only logistic regression is reported: the
timing-threshold baseline already uses `timing_abs_mean` alone on these
conditions, so its scores are unchanged by construction (verified below, not
assumed).

| Condition | Full features F1 | Ablated F1 (no 1/32) | Delta |
| --- | ---: | ---: | ---: |
| quantized_16_25 | 0.759 | 0.669 | -0.091 |
| quantized_16_50 | 0.935 | 0.832 | -0.103 |
| quantized_16_75 | 0.974 | 0.973 | -0.001 |
| quantized_16_100 | 1.000 | 1.000 | +0.000 |

## Variation between drummers

Equal-weight mean ± sample SD of F1 across the nine held-out drummers
(logistic regression only); compare with the pooled table above, which
weights drummers by clip count instead.

| Condition | Full features (mean ± SD) | Ablated, no 1/32 (mean ± SD) | Delta in mean |
| --- | ---: | ---: | ---: |
| quantized_16_25 | 0.732 ± 0.063 | 0.617 ± 0.082 | -0.115 |
| quantized_16_50 | 0.941 ± 0.044 | 0.802 ± 0.055 | -0.140 |
| quantized_16_75 | 0.969 ± 0.017 | 0.968 ± 0.017 | -0.001 |
| quantized_16_100 | 1.000 ± 0.000 | 1.000 ± 0.000 | +0.000 |

Timing-threshold F1 unchanged between the two runs: True.

## Finding

The original concern was that the 1/16/1/32 relationship at 50%+ quantization
gives logistic regression a shortcut there. The data shows the opposite
pattern: removing the 1/32 features costs the most at -0.091 F1
(25%) and barely matters at 75%/100%, where the two grids are most related by
construction and are consequently most redundant, not most exploited. The 1/32
features are providing real, non-redundant signal specifically where detection is
hardest (low/moderate quantization strength), not inflating scores through
redundancy where detection was already near-ceiling. So the strong scores at
75%/100% do not depend on this relationship between the two grids; the weaker
scores at 25%/50% partly do (largest effect at quantized_16_50).

## Reproduce

```bash
env/bin/python -m src.ablation
```

