"""Run the fixed baseline protocol and save small shareable result artifacts."""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.common.config import FEATURES_DIR, SEED
from src.models.baseline import (
    EXPERIMENT_CONDITIONS, FEATURE_COLUMNS, MODEL_NAMES,
    evaluate_table, summarize_metrics,
)

MODEL_LABELS = {"logistic_regression": "Logistic regression", "timing_threshold": "Timing threshold"}
CONDITION_LABELS = {
    "quantized_16_25": "1/16 at 25%", "quantized_16_50": "1/16 at 50%",
    "quantized_16_75": "1/16 at 75%", "quantized_16_100": "1/16 at 100%",
    "quantized_32_100": "1/32 at 100%", "velocity_normalized": "Velocity only",
}


def plot_results(summary: pd.DataFrame, folds: pd.DataFrame, output: Path) -> None:
    """Use separate plots for strength, drummer spread, and pooled confusion."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "groove-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "groove-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strength_conditions = list(EXPERIMENT_CONDITIONS[:4])
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for model, color in zip(MODEL_NAMES, ("#2865a2", "#c27935")):
        rows = summary[summary.model == model].set_index("condition").loc[strength_conditions]
        ax.errorbar(
            [25, 50, 75, 100], rows.drummer_mean_f1, yerr=rows.drummer_sd_f1,
            marker="o", capsize=4, label=MODEL_LABELS[model], color=color,
        )
    ax.set(
        xlabel="Movement toward the 1/16 grid (%)", ylabel="F1 for transformed clips",
        title="Detection across held-out drummers", xticks=[25, 50, 75, 100], ylim=(-0.08, 1.08),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.text(0.5, -0.025, "Equal-weight mean across drummers; bars show sample SD, not confidence intervals.", ha="center", fontsize=9)
    fig.savefig(output / "f1_by_strength.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    drummers = sorted(folds.test_drummer.unique(), key=lambda name: int(name.removeprefix("drummer")))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained", sharey=True)
    for ax, model in zip(axes, MODEL_NAMES):
        values = folds[folds.model == model].pivot(index="condition", columns="test_drummer", values="f1")
        values = values.reindex(index=EXPERIMENT_CONDITIONS, columns=drummers)
        plot = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(values)):
            for j in range(len(drummers)):
                value = values.iloc[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.6 else "black")
        ax.set(
            title=MODEL_LABELS[model], xlabel="Held-out drummer ID", xticks=range(len(drummers)),
            xticklabels=[name.removeprefix("drummer") for name in drummers],
            yticks=range(len(EXPERIMENT_CONDITIONS)),
            yticklabels=[CONDITION_LABELS[c] for c in EXPERIMENT_CONDITIONS],
        )
    fig.colorbar(plot, ax=axes, label="F1 for transformed clips", shrink=0.8)
    fig.savefig(output / "f1_by_drummer.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(19, 7.1), layout="constrained")
    for row_index, model in enumerate(MODEL_NAMES):
        for column_index, condition in enumerate(EXPERIMENT_CONDITIONS):
            ax = axes[row_index, column_index]
            record = summary[(summary.condition == condition) & (summary.model == model)].iloc[0]
            counts = np.array([[record.tn, record.fp], [record.fn, record.tp]], dtype=int)
            proportions = counts / counts.sum(axis=1, keepdims=True)
            ax.imshow(proportions, vmin=0, vmax=1, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{counts[i, j]:,}\n{proportions[i, j]:.1%}", ha="center", va="center", fontsize=9, color="white" if proportions[i, j] > 0.6 else "black")
            ax.set(xticks=[0, 1], xticklabels=["Human", "Edited"], yticks=[0, 1], yticklabels=["Human", "Edited"], xlabel="Predicted")
            if row_index == 0:
                ax.set_title(CONDITION_LABELS[condition], fontsize=10)
            if column_index == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nActual")
    fig.suptitle("Pooled held-out predictions: counts and percentages within each actual class", fontsize=14)
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def write_report(summary: pd.DataFrame, folds: pd.DataFrame, config: dict, output: Path) -> None:
    """Keep results descriptive; no reliability cutoff was chosen for this run."""
    lines = [
        "# Baseline classifier results", "",
        f"Run: {config['run_at']}.", "",
        "## Method", "",
        "Six separate balanced binary experiments compare original clips with one edited condition at a time. Edited clips are the positive class (1); human clips are class 0.", "",
        f"Each experiment has {int(summary.n_test.iloc[0]):,} clips. Leave-one-drummer-out evaluation trains on {config['n_drummers'] - 1} drummers and tests on the remaining drummer, repeating for all {config['n_drummers']} drummers. Every row is tested exactly once per experiment and method. All versions of a clip, and all clips from a performance, stay with their drummer.", "",
        "- Logistic regression uses the 10 existing numerical features, StandardScaler fitted only on the training fold, and fixed settings: C=1, lbfgs, max_iter=2000, seed=42. The probability decision threshold is 0.5. No hyperparameter search or test-driven feature selection was performed.",
        "- The timing rule predicts an edit when mean absolute grid deviation is at or below a cutoff chosen only from training rows. The cutoff maximizes balanced accuracy; tied objectives choose the lowest cutoff. It uses the 1/16 deviation for the four 1/16 experiments and the velocity-only control, and the 1/32 deviation for the 1/32 experiment.",
        "- Both methods receive only rhythmic features. IDs, drummer labels, original GMD split labels, condition names, and transformation settings are excluded from classifier inputs.",
        "- These drummer-held-out folds replace the supplied GMD split for this experiment. There is no additional untouched test cohort; these results evaluate the fixed baseline protocol.", "",
        "## Equal-weight scores across drummers", "",
        "Each drummer contributes equally below. Values are mean ± sample standard deviation across held-out drummers; this spread is not a confidence interval. Precision, recall, and F1 refer to edited clips. Precision and F1 are recorded as 0 when no positive clips are predicted.", "",
        "| Condition | Method | Accuracy | Precision | Recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in EXPERIMENT_CONDITIONS:
        for model in MODEL_NAMES:
            row = summary[(summary.condition == condition) & (summary.model == model)].iloc[0]
            values = [f"{row[f'drummer_mean_{m}']:.3f} ± {row[f'drummer_sd_{m}']:.3f}" for m in ("accuracy", "precision", "recall", "f1")]
            lines.append(f"| {CONDITION_LABELS[condition]} | {MODEL_LABELS[model]} | " + " | ".join(values) + " |")
    lines += [
        "", "## Pooled scores", "",
        "These metrics pool all held-out predictions. Drummers with more clips have more influence here; compare with the equal-weight scores above.", "",
        "| Condition | Method | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {CONDITION_LABELS[row.condition]} | {MODEL_LABELS[row.model]} | "
            f"{row.accuracy:.3f} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | "
            f"{row.tn:,} | {row.fp:,} | {row.fn:,} | {row.tp:,} |"
        )
    lines += [
        "", "## Figures", "",
        "![Detection versus 1/16 quantization strength](f1_by_strength.png)", "",
        "The 1/32 grid and velocity-only condition are separate experiments and are not points on the 1/16 strength curve.", "",
        "![F1 for each held-out drummer](f1_by_drummer.png)", "",
        "![Confusion matrices](confusion_matrices.png)", "",
        "## Checks and interpretation", "",
        "- All source clips have exactly one row for each of the seven conditions. Features are finite and matching versions have consistent drummer, source file, and note count.",
        "- Training and test drummers, source performances, and source clip IDs are disjoint in every fold. Stored held-out predictions have exactly one row per clip, method, and experiment.",
        "- The velocity-only timing control must have 50% accuracy because each human/edited pair has identical timing. Its F1 need not be 0.5: F1 depends on which class receives the predictions, and a tied training cutoff can yield no positive predictions.",
        "- Full grid snapping and constant velocity leave direct signatures in the selected features. At 50% or stronger movement to the 1/16 grid, a geometric relation between the 1/16 and 1/32 deviation features can also help the classifier. High scores measure detection of these synthetic edits, not general authorship or listener judgments.",
        "- Results apply to the retained two-bar GMD subset, including straight, swung, and triplet patterns. The original short-clip exclusions and uneven drummer coverage still apply.",
        "- No formal numerical reliability criterion was fixed for this run. Scores and their drummer-to-drummer variation are reported without declaring a universal detectability threshold.", "",
        "## Reproduce and inspect", "",
        "From the project root:", "", "```bash", "env/bin/python -m src.train_baseline", "```", "",
        "- `metrics.csv`: pooled metrics plus equal-weight drummer means and SDs.",
        "- `fold_metrics.csv`: each held-out drummer, training drummer IDs, learned timing cutoff, and confusion counts.",
        "- `run_config.json`: fixed settings, input-table SHA-256, feature list, and package versions.",
        f"- `{config['predictions_output']}` in the project: full held-out labels/predictions, kept outside Git with other generated data when using the default path.", "",
        f"Input SHA-256: `{config['input_sha256']}`.", "",
    ]
    (output / "MODEL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FEATURES_DIR / "conditions_features.csv")
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    parser.add_argument("--predictions-output", type=Path, default=FEATURES_DIR / "baseline_predictions.csv")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Feature table not found: {args.input}")
    frame = pd.read_csv(args.input)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        folds, predictions = evaluate_table(frame)
    expected_pairs = pd.MultiIndex.from_product(
        [EXPERIMENT_CONDITIONS, MODEL_NAMES], names=["condition", "model"]
    )
    prediction_groups = predictions.groupby(["condition", "model"]).size()
    if not prediction_groups.index.sort_values().equals(expected_pairs.sort_values()):
        raise ValueError("Missing experiment/method combinations")
    expected_folds = pd.MultiIndex.from_product(
        [EXPERIMENT_CONDITIONS, MODEL_NAMES, frame.drummer.unique()],
        names=["condition", "model", "test_drummer"],
    )
    observed_folds = pd.MultiIndex.from_frame(folds[["condition", "model", "test_drummer"]])
    if not observed_folds.sort_values().equals(expected_folds.sort_values()):
        raise ValueError("Missing or repeated held-out drummer folds")
    if predictions.duplicated(["condition", "model", "clip_id"]).any():
        raise ValueError("Repeated held-out predictions")
    expected = frame[frame.condition == "human"].shape[0] * 2
    if not prediction_groups.eq(expected).all():
        raise ValueError("Missing held-out predictions")
    control = folds[(folds.condition == "velocity_normalized") & (folds.model == "timing_threshold")]
    if not np.allclose(control.accuracy, 0.5, rtol=0, atol=1e-12):
        raise ValueError("The velocity-only timing control should have exactly 50% accuracy")
    summary = summarize_metrics(folds, predictions)
    config = {
        "run_at": datetime.now(ZoneInfo("Europe/Stockholm")).isoformat(timespec="seconds"),
        "input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "predictions_output": str(args.predictions_output),
        "source_rows": len(frame), "n_drummers": int(frame.drummer.nunique()),
        "features": list(FEATURE_COLUMNS), "experiments": list(EXPERIMENT_CONDITIONS),
        "positive_class": "transformed", "evaluation": "leave-one-drummer-out",
        "logistic_regression": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, "random_state": SEED, "decision_threshold": 0.5},
        "preprocessing": "StandardScaler fitted on training fold only",
        "timing_threshold": {"objective": "training balanced accuracy", "direction": "deviation <= cutoff", "tie": "lowest cutoff within 1e-12 objective tolerance"},
        "hyperparameter_search": False, "formal_reliability_criterion": None,
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")}},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output / "fold_metrics.csv", index=False)
    summary.to_csv(args.output / "metrics.csv", index=False)
    predictions.to_csv(args.predictions_output, index=False)
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    plot_results(summary, folds, args.output)
    write_report(summary, folds, config, args.output)
    print(summary[["condition", "model", "accuracy", "f1", "drummer_mean_f1", "drummer_sd_f1"]].to_string(index=False))
    print(f"Saved {len(folds)} fold rows and {len(predictions):,} held-out predictions.")
    print(f"Summary: {args.output / 'MODEL_RESULTS.md'}")


if __name__ == "__main__":
    main()
