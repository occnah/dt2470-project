"""
Qualitative error analysis: where do the models actually get it wrong,
broken down by drummer, style, and tempo?

This does not retrain anything -- it reads the feature table and the full
held-out predictions already saved by `src.run_experiments`, joins each
prediction with its clip's drummer/style/tempo, and reports false-positive
and false-negative rates per group, plus concrete example clips for the
worst groups so they can be looked up in the original MIDI (or exported with
`src.build_dataset --midi-output` for listening).

A representative condition is picked from each part of the project rather
than all 27, so the report stays readable: one clearly below the reliability
threshold, one right at it, and one from each of the three research
questions.

Run after `src.build_dataset` and `src.run_experiments` (both needed once,
not per analysis):

    env/bin/python -m src.error_analysis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import FEATURES_DIR, FEATURES_TABLE, HUMAN, RESULTS_DIR

DEFAULT_CONDITIONS = (
    "quantize_16_25",            # clearly below the reliability threshold
    "quantize_16_75",            # right at the reliability threshold
    "velocity_25",               # RQ1, below threshold
    "humanize_pink_matched",     # RQ2/RQ3, hardest humanization condition
    "deviations_shuffled",       # RQ3 control
)
DEFAULT_MODEL = "logistic_regression"
TEMPO_BINS = (0, 90, 110, 130, 150, np.inf)
TEMPO_LABELS = ("<90", "90-110", "110-130", "130-150", "150+")
MIN_GROUP_SIZE = 20  # ignore groups with too few clips for a stable rate
N_EXAMPLES = 5


def clip_metadata(features: pd.DataFrame) -> pd.DataFrame:
    """
    One row per source clip: style, tempo, tempo bucket (from the human
    version). Drummer is not included here -- `predictions.csv` already has
    it, and joining on it too would just duplicate the column.
    """
    human = features.loc[features.condition == HUMAN, ["source_clip_id", "style", "tempo"]].copy()
    human["tempo_bucket"] = pd.cut(human.tempo, TEMPO_BINS, labels=TEMPO_LABELS, right=False)
    return human.set_index("source_clip_id")


def error_rates(rows: pd.DataFrame, by: str) -> pd.DataFrame:
    """
    False-positive rate (human clips wrongly called edited) and false-negative
    rate (edited clips wrongly called human) per value of `by`, plus group size.
    Groups smaller than MIN_GROUP_SIZE are dropped as too noisy to read.
    """
    out = []
    for value, group in rows.groupby(by, observed=True):
        human_rows = group[group.y_true == 0]
        edited_rows = group[group.y_true == 1]
        if len(human_rows) < MIN_GROUP_SIZE or len(edited_rows) < MIN_GROUP_SIZE:
            continue
        out.append({
            by: value,
            "n_clips": len(group) // 2,
            "false_positive_rate": float((human_rows.y_pred == 1).mean()),
            "false_negative_rate": float((edited_rows.y_pred == 0).mean()),
        })
    return pd.DataFrame(out).sort_values("false_negative_rate", ascending=False)


def worst_examples(rows: pd.DataFrame, by: str, value, kind: str, n: int = N_EXAMPLES) -> pd.DataFrame:
    """Up to `n` example clips of the given error kind ('fp' or 'fn') in group `value` of `by`."""
    group = rows[rows[by] == value]
    if kind == "fp":
        wrong = group[(group.y_true == 0) & (group.y_pred == 1)]
    else:
        wrong = group[(group.y_true == 1) & (group.y_pred == 0)]
    return wrong[["source_clip_id", "drummer", "style", "tempo"]].drop_duplicates().head(n)


def write_report(
    predictions: pd.DataFrame,
    meta: pd.DataFrame,
    conditions: tuple[str, ...],
    model: str,
    output: Path,
) -> None:
    lines = [
        "# Qualitative error analysis", "",
        f"Model: `{model}`. Conditions below are a representative subset, not all 27 -- ",
        "see `results/experiments/RESULTS.md` for every condition's pooled scores. ",
        "False-positive rate is over human clips only (wrongly called edited); ",
        "false-negative rate is over edited clips only (wrongly called human). ",
        f"Groups with fewer than {MIN_GROUP_SIZE} clips on either side are omitted ",
        "as too small to read a rate from.", "",
    ]

    rows = predictions[(predictions.model == model) & (predictions.condition.isin(conditions))].copy()
    rows = rows.join(meta, on="source_clip_id")
    missing = rows.drummer.isna().sum()
    if missing:
        rows = rows.dropna(subset=["drummer"])

    for condition in conditions:
        condition_rows = rows[rows.condition == condition]
        if condition_rows.empty:
            lines += [f"## {condition}", "", "No predictions found for this condition in `predictions.csv`.", ""]
            continue

        lines += [f"## {condition}", ""]
        overall_fp = (condition_rows[condition_rows.y_true == 0].y_pred == 1).mean()
        overall_fn = (condition_rows[condition_rows.y_true == 1].y_pred == 0).mean()
        lines.append(f"Overall: false-positive rate {overall_fp:.3f}, false-negative rate {overall_fn:.3f}.")
        lines.append("")

        for by, heading in (("drummer", "By drummer"), ("style", "By style"), ("tempo_bucket", "By tempo")):
            table = error_rates(condition_rows, by)
            if table.empty:
                continue
            lines += [f"### {heading}", "", f"| {by.replace('_', ' ').title()} | Clips | FP rate | FN rate |",
                      "| --- | ---: | ---: | ---: |"]
            for _, row in table.iterrows():
                lines.append(f"| {row[by]} | {row.n_clips} | {row.false_positive_rate:.3f} | {row.false_negative_rate:.3f} |")
            lines.append("")

            worst = table.iloc[0]
            if worst.false_negative_rate > 0:
                examples = worst_examples(condition_rows, by, worst[by], "fn")
                if not examples.empty:
                    lines += [
                        f"Worst {by.replace('_', ' ')} for false negatives ({worst[by]}, "
                        f"rate {worst.false_negative_rate:.3f}) -- example clips classified as human "
                        f"despite being `{condition}`:", "",
                    ]
                    for _, example in examples.iterrows():
                        lines.append(
                            f"- `{example.source_clip_id}` (drummer {example.drummer}, "
                            f"style {example.style}, tempo {example.tempo:.0f})"
                        )
                    lines.append("")

    lines += [
        "## How to listen to an example",
        "",
        "Re-run the build with `--midi-output data/generated` to export every edited ",
        "performance as MIDI, then find the listed `source_clip_id`'s source file under ",
        f"`data/generated/{conditions[0]}/<source_file>` (human version is the original file ",
        "in `data/groove/`) and compare by ear around the clip's start/end time in ",
        "`data/features/features.csv`.", "",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "ERROR_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=FEATURES_TABLE)
    parser.add_argument("--predictions", type=Path, default=FEATURES_DIR / "predictions.csv")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "error_analysis")
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    for path in (args.features, args.predictions):
        if not path.is_file():
            parser.error(f"Not found: {path} (run src.build_dataset and src.run_experiments first)")

    features = pd.read_csv(args.features)
    predictions = pd.read_csv(args.predictions)
    meta = clip_metadata(features)

    write_report(predictions, meta, tuple(args.conditions), args.model, args.output)
    print(f"Saved: {args.output / 'ERROR_ANALYSIS.md'}")


if __name__ == "__main__":
    main()
