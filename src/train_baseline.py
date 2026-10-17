"""
Baseline classifier: human vs. a quantized/velocity-normalized condition.

Drummer-disjoint split (never splits a single performance's clips across
train/test), magnitude features only, two simple baselines (logistic
regression and a random forest) so we can see whether the signal is there
before trying anything fancier.

Run from the project root, with --conditions features already built:

    env/bin/python -m src.train_baseline
    env/bin/python -m src.train_baseline --condition quantized_16_50
    env/bin/python -m src.train_baseline --condition velocity_normalized
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import FEATURES_DIR, HUMAN, SEED

# Magnitude features only. Identifiers, condition metadata (subdivision,
# strength, normalized_velocity), and tempo/clip bounds are deliberately
# excluded from the classifier inputs -- see CODE_EXPLAINER.md.
FEATURE_COLUMNS = [
    "note_count",
    "timing_abs_mean",
    "timing_std",
    "ioi_mean",
    "ioi_std",
    "velocity_mean",
    "velocity_std",
    "velocity_range",
    "timing_32_abs_mean",
    "timing_32_std",
]


def load_binary_task(path: Path, condition: str) -> pd.DataFrame:
    """Keep only human rows and one other condition, matched by source_clip_id."""
    df = pd.read_csv(path)
    available = sorted(df["condition"].unique())
    if condition not in available:
        raise SystemExit(f"Unknown condition {condition!r}. Available: {available}")

    task = df[df["condition"].isin([HUMAN, condition])].copy()
    task["label"] = (task["condition"] != HUMAN).astype(int)  # 0 = human, 1 = other
    return task


def drummer_disjoint_split(task: pd.DataFrame, test_size: float, seed: int):
    """Split by drummer so no performance's clips appear in both train and test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(task, groups=task["drummer"]))
    return task.iloc[train_idx], task.iloc[test_idx]


def impute_missing(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]):
    """Fill NaNs (e.g. single-onset clips with no IOI) with the train median."""
    medians = train[columns].median()
    return train[columns].fillna(medians), test[columns].fillna(medians)


def report(name: str, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["human", "other"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FEATURES_DIR / "conditions_features.csv")
    parser.add_argument(
        "--condition", default="quantized_16_100",
        help="Condition to distinguish from human (default: quantized_16_100, the easiest case).",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of drummers held out for test.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(
            f"{args.input} not found. Build it first:\n"
            f"  env/bin/python -m src.build_dataset --conditions"
        )

    task = load_binary_task(args.input, args.condition)
    train, test = drummer_disjoint_split(task, args.test_size, args.seed)

    print(f"Task: human vs. {args.condition}")
    print(f"Train: {len(train)} rows, drummers {sorted(train['drummer'].unique())}")
    print(f"Test:  {len(test)} rows, drummers {sorted(test['drummer'].unique())}")
    if set(train["drummer"]) & set(test["drummer"]):
        raise AssertionError("Drummer leaked across the train/test split")

    X_train, X_test = impute_missing(train, test, FEATURE_COLUMNS)
    y_train, y_test = train["label"].to_numpy(), test["label"].to_numpy()

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
    logreg.fit(X_train, y_train)
    report("Logistic regression", y_test, logreg.predict(X_test))
    coefs = pd.Series(logreg.named_steps["logisticregression"].coef_[0], index=FEATURE_COLUMNS)
    print("Standardized coefficients (sign shows direction, magnitude shows weight):")
    print(coefs.sort_values(key=abs, ascending=False).round(3).to_string())

    forest = RandomForestClassifier(n_estimators=200, random_state=args.seed)
    forest.fit(X_train, y_train)
    report("Random forest", y_test, forest.predict(X_test))
    importances = pd.Series(forest.feature_importances_, index=FEATURE_COLUMNS)
    print("Feature importances:")
    print(importances.sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
