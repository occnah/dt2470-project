"""Feature ablation: do the 1/32 timing features give logistic regression a shortcut?

WHATS_LEFT.md's highest-priority open item: at 50% or stronger movement to
the 1/16 grid, `timing_32_abs_mean`/`timing_32_std` become related to the
1/16 deviation features by construction, which could let logistic
regression detect quantization through that relationship rather than
genuine timing deviation. This reruns the four 1/16 experiments
(quantized_16_25/50/75/100) with the 1/32 features removed from the
classifier's inputs, using the same leave-one-drummer-out protocol as
results/baseline/, and compares F1 against the full-feature run.

Only logistic regression can change here: the timing-threshold baseline
already uses `timing_abs_mean` alone on these four conditions (never the
1/32 features), so its scores are identical in both runs by construction
and are reported only as a sanity check.

Run from the project root, after `src.build_dataset --conditions`:

    env/bin/python -m src.ablation
"""

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import FEATURES_DIR
from src.models.baseline import (
    EXPERIMENT_CONDITIONS, FEATURE_COLUMNS, evaluate_table, summarize_metrics,
)

SIXTEENTH_CONDITIONS = tuple(c for c in EXPERIMENT_CONDITIONS if c.startswith("quantized_16_"))
DROPPED_FEATURES = ("timing_32_abs_mean", "timing_32_std")
ABLATED_FEATURE_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in DROPPED_FEATURES)


def run(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_summary, ablated_summary) over the four 1/16 conditions."""
    full_folds, full_predictions = evaluate_table(frame, conditions=SIXTEENTH_CONDITIONS)
    full_summary = summarize_metrics(full_folds, full_predictions, conditions=SIXTEENTH_CONDITIONS)

    ablated_folds, ablated_predictions = evaluate_table(
        frame, conditions=SIXTEENTH_CONDITIONS, feature_columns=ABLATED_FEATURE_COLUMNS,
    )
    ablated_summary = summarize_metrics(ablated_folds, ablated_predictions, conditions=SIXTEENTH_CONDITIONS)
    return full_summary, ablated_summary


def write_report(full_summary: pd.DataFrame, ablated_summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# Feature ablation: 1/16 detection without the 1/32 timing features", "",
        f"Dropped features: `{'`, `'.join(DROPPED_FEATURES)}`.",
        f"Kept features: `{'`, `'.join(ABLATED_FEATURE_COLUMNS)}`.", "",
        "Same leave-one-drummer-out protocol as `results/baseline/`, restricted to the",
        "four 1/16 quantization conditions. Only logistic regression is reported: the",
        "timing-threshold baseline already uses `timing_abs_mean` alone on these",
        "conditions, so its scores are unchanged by construction (verified below, not",
        "assumed).", "",
        "| Condition | Full features F1 | Ablated F1 (no 1/32) | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition in SIXTEENTH_CONDITIONS:
        full_f1 = full_summary.loc[
            (full_summary.condition == condition) & (full_summary.model == "logistic_regression"), "f1"
        ].iloc[0]
        ablated_f1 = ablated_summary.loc[
            (ablated_summary.condition == condition) & (ablated_summary.model == "logistic_regression"), "f1"
        ].iloc[0]
        lines.append(f"| {condition} | {full_f1:.3f} | {ablated_f1:.3f} | {ablated_f1 - full_f1:+.3f} |")

    full_threshold = full_summary.loc[full_summary.model == "timing_threshold", "f1"].to_numpy()
    ablated_threshold = ablated_summary.loc[ablated_summary.model == "timing_threshold", "f1"].to_numpy()
    unchanged = bool((full_threshold == ablated_threshold).all())
    lines += [
        "",
        f"Timing-threshold F1 unchanged between the two runs: {unchanged}.", "",
        "## Interpretation", "",
        "A small or no F1 drop when the 1/32 features are removed means logistic",
        "regression was not relying on their relationship to the 1/16 deviation —",
        "the 1/16 deviation features alone already carry the detectable signal. A",
        "large drop, concentrated at 50% and above, would mean the full-feature",
        "result partly reflected that construction-level relationship rather than",
        "detection that would hold with 1/16 features alone.", "",
        "## Reproduce", "", "```bash", "env/bin/python -m src.ablation", "```", "",
    ]
    (output / "ABLATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FEATURES_DIR / "conditions_features.csv")
    parser.add_argument("--output", type=Path, default=Path("results/ablation"))
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Feature table not found: {args.input}")
    frame = pd.read_csv(args.input)

    full_summary, ablated_summary = run(frame)

    args.output.mkdir(parents=True, exist_ok=True)
    full_summary.to_csv(args.output / "full_metrics.csv", index=False)
    ablated_summary.to_csv(args.output / "ablated_metrics.csv", index=False)
    write_report(full_summary, ablated_summary, args.output)

    comparison = pd.DataFrame({
        "condition": SIXTEENTH_CONDITIONS,
        "full_f1": [
            full_summary.loc[(full_summary.condition == c) & (full_summary.model == "logistic_regression"), "f1"].iloc[0]
            for c in SIXTEENTH_CONDITIONS
        ],
        "ablated_f1": [
            ablated_summary.loc[(ablated_summary.condition == c) & (ablated_summary.model == "logistic_regression"), "f1"].iloc[0]
            for c in SIXTEENTH_CONDITIONS
        ],
    })
    comparison["delta"] = comparison.ablated_f1 - comparison.full_f1
    print(comparison.to_string(index=False))
    print(f"Saved: {args.output / 'ABLATION_RESULTS.md'}")


if __name__ == "__main__":
    main()
