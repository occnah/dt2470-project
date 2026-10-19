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

Timing-threshold F1 unchanged between the two runs: True.

## Interpretation

A small or no F1 drop when the 1/32 features are removed means logistic
regression was not relying on their relationship to the 1/16 deviation —
the 1/16 deviation features alone already carry the detectable signal. A
large drop, concentrated at 50% and above, would mean the full-feature
result partly reflected that construction-level relationship rather than
detection that would hold with 1/16 features alone.

## Reproduce

```bash
env/bin/python -m src.ablation
```

