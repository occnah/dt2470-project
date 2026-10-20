"""
RQ3 feature analysis: which feature groups carry the signal?

Reruns the leave-one-drummer-out evaluation with different feature sets:

- all              every feature
- magnitude only   context + `mag_*`
- structure only   context + `struct_*`
- without <group>  every feature except one group (see FEATURE_GROUPS)

and saves results/ablation/ (ABLATION.md, ablation_metrics.csv, figures,
and logistic-regression coefficients from the full feature set).

    env/bin/python -m src.run_ablation
    env/bin/python -m src.run_ablation --models logistic_regression --conditions humanize_pink_matched
    env/bin/python -m src.run_ablation --report-only   # redraw report/figures from saved CSVs
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.common.config import FEATURES_TABLE, RESULTS_DIR
from src.data.conditions import BY_NAME, EDITED_CONDITIONS
from src.features import (
    CONTEXT_FEATURES, FEATURE_GROUPS, MAGNITUDE_COLUMNS, MODEL_INPUTS, STRUCTURE_COLUMNS,
)
from src.models.classifiers import MODELS_BY_NAME
from src.models.evaluation import evaluate, summarize
from src.reporting.ablation_report import write_ablation_report

DEFAULT_MODELS = ("logistic_regression", "random_forest")


def feature_sets() -> dict[str, tuple[str, ...]]:
    """Name -> model input columns, in report order."""
    sets = {
        "all": MODEL_INPUTS,
        "magnitude only": CONTEXT_FEATURES + MAGNITUDE_COLUMNS,
        "structure only": CONTEXT_FEATURES + STRUCTURE_COLUMNS,
    }
    for group, columns in FEATURE_GROUPS.items():
        sets[f"without {group}"] = tuple(c for c in MODEL_INPUTS if c not in columns)
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=FEATURES_TABLE)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ablation")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        choices=[m for m in MODELS_BY_NAME if m != "timing_rule"])
    parser.add_argument("--conditions", nargs="+", choices=[c.name for c in EDITED_CONDITIONS], help="Default: all")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild ABLATION.md and figures from the CSVs in --output without retraining")
    args = parser.parse_args()
    if args.report_only:
        summary = pd.read_csv(args.output / "ablation_metrics.csv")
        coefficients = pd.read_csv(args.output / "coefficients.csv")
        models = tuple(MODELS_BY_NAME[m] for m in summary.model.unique())
        conditions = tuple(c for c in EDITED_CONDITIONS if c.name in set(summary.condition))
        write_ablation_report(summary, coefficients, list(feature_sets()), conditions, models, args.output)
        print(f"Rebuilt {args.output / 'ABLATION.md'}")
        return
    if not args.input.is_file():
        parser.error(f"Feature table not found: {args.input} (run src.build_dataset first)")

    frame = pd.read_csv(args.input)
    models = tuple(MODELS_BY_NAME[m] for m in args.models)
    conditions = tuple(BY_NAME[c] for c in args.conditions) if args.conditions else EDITED_CONDITIONS

    summaries, coefficients = [], None
    with warnings.catch_warnings(), threadpool_limits(limits=1, user_api="blas"):
        warnings.simplefilter("error", ConvergenceWarning)
        for name, columns in feature_sets().items():
            print(f"Feature set: {name} ({len(columns)} columns)", flush=True)
            result = evaluate(frame, conditions, models, feature_columns=columns, verbose=False)
            summaries.append(summarize(result).assign(feature_set=name))
            if name == "all":
                coefficients = result.coefficients
    summary = pd.concat(summaries, ignore_index=True)

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "ablation_metrics.csv", index=False)
    coefficients.to_csv(args.output / "coefficients.csv", index=False)
    write_ablation_report(summary, coefficients, list(feature_sets()), conditions, models, args.output)
    print(f"Saved {args.output / 'ABLATION.md'}")


if __name__ == "__main__":
    main()
