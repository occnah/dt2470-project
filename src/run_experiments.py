"""
Run every model on every edited condition (leave-one-drummer-out) and save
the results to results/experiments/.

    env/bin/python -m src.run_experiments
    env/bin/python -m src.run_experiments --models logistic_regression --conditions quantize_16_25
    env/bin/python -m src.run_experiments --report-only   # redraw report/figures from saved CSVs

Outputs: RESULTS.md, metrics.csv (one row per condition and model),
fold_metrics.csv (per held-out drummer), reliability.csv, coefficients.csv
(logistic regression), run_config.json and figures. Full held-out
predictions go to data/features/predictions.csv (not in git).
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.common.config import FEATURES_DIR, FEATURES_TABLE, RELIABLE_F1, RESULTS_DIR, SEED
from src.data.conditions import BY_NAME, EDITED_CONDITIONS
from src.features import MODEL_INPUTS
from src.models.classifiers import MODELS, MODELS_BY_NAME
from src.models.evaluation import evaluate, reliability_thresholds, summarize
from src.reporting.experiment_report import write_experiment_report
from src.reporting.figures import plot_f1_by_drummer, plot_humanization, plot_quantization


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_config(frame: pd.DataFrame, path: Path, conditions, models) -> dict:
    """Everything needed to tell what this run did."""
    return {
        "run_at": datetime.now(ZoneInfo("Europe/Stockholm")).isoformat(timespec="seconds"),
        "input": str(path), "input_sha256": file_sha256(path),
        "n_source_clips": int(frame.source_clip_id.nunique()),
        "n_drummers": int(frame.drummer.nunique()),
        "conditions": [c.name for c in conditions],
        "models": [m.name for m in models],
        "model_inputs": list(MODEL_INPUTS),
        "evaluation": "leave-one-drummer-out, balanced human vs edited per condition",
        "reliable_f1": RELIABLE_F1, "seed": SEED, "hyperparameter_search": False,
        "versions": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name)
               for name in ("numpy", "pandas", "scikit-learn", "pretty_midi")},
        },
    }


def write_outputs(summary, folds, thresholds, config, models, output: Path, complete: bool = True) -> None:
    """Figures and RESULTS.md. The report and curves need every condition."""
    labelled = [(m.name, m.label) for m in models]
    if complete:
        plot_quantization(summary, labelled, output / "quantization_curves.png")
        plot_humanization(summary, labelled, output / "humanization_curves.png")
        write_experiment_report(summary, thresholds, config, labelled, output)
    plot_f1_by_drummer(folds, labelled, output / "f1_by_drummer.png")


def write_outputs_from_saved(output: Path) -> None:
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    models = tuple(MODELS_BY_NAME[name] for name in config["models"])
    complete = set(config["conditions"]) == {c.name for c in EDITED_CONDITIONS}
    write_outputs(
        pd.read_csv(output / "metrics.csv"), pd.read_csv(output / "fold_metrics.csv"),
        pd.read_csv(output / "reliability.csv"), config, models, output, complete,
    )
    print(f"Rebuilt report and figures in {output}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=FEATURES_TABLE)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "experiments")
    parser.add_argument("--predictions", type=Path, default=FEATURES_DIR / "predictions.csv")
    parser.add_argument("--models", nargs="+", choices=list(MODELS_BY_NAME), help="Default: all")
    parser.add_argument("--conditions", nargs="+", choices=[c.name for c in EDITED_CONDITIONS], help="Default: all")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild RESULTS.md and figures from the CSVs in --output without retraining")
    args = parser.parse_args()
    if args.report_only:
        write_outputs_from_saved(args.output)
        return
    if not args.input.is_file():
        parser.error(f"Feature table not found: {args.input} (run src.build_dataset first)")

    models = tuple(MODELS_BY_NAME[m] for m in args.models) if args.models else MODELS
    conditions = tuple(BY_NAME[c] for c in args.conditions) if args.conditions else EDITED_CONDITIONS
    frame = pd.read_csv(args.input)
    print(f"{len(conditions)} conditions x {len(models)} models x {frame.drummer.nunique()} held-out drummers")

    with warnings.catch_warnings(), threadpool_limits(limits=1, user_api="blas"):
        warnings.simplefilter("error", ConvergenceWarning)
        result = evaluate(frame, conditions, models)
    summary = summarize(result)
    thresholds = reliability_thresholds(summary)
    config = run_config(frame, args.input, conditions, models)

    args.output.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "metrics.csv", index=False)
    result.folds.to_csv(args.output / "fold_metrics.csv", index=False)
    result.coefficients.to_csv(args.output / "coefficients.csv", index=False)
    thresholds.to_csv(args.output / "reliability.csv", index=False)
    result.predictions.to_csv(args.predictions, index=False)
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    write_outputs(summary, result.folds, thresholds, config, models, args.output,
                  complete=not args.conditions)

    print(summary[["condition", "model", "drummer_mean_f1", "drummer_min_f1", "reliable"]].to_string(index=False))
    print(f"Saved results to {args.output}/ and predictions to {args.predictions}")


if __name__ == "__main__":
    main()
