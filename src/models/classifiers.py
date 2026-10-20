"""
The classifiers compared in every experiment.

`MODELS` is the single list of models. All settings are fixed in advance
(no hyperparameter search), so no test data can influence them. Every
model fills undefined feature values with the training-fold median, and
scaling (where used) is fitted on the training fold only: both happen
inside the pipeline, which is refitted for every held-out drummer.

- logistic_regression  linear model on standardized features
- random_forest        nonlinear, unscaled features
- svm                  RBF-kernel SVM on standardized features
- timing_rule          reference: one threshold on mean absolute timing
                       deviation (1/32 grid for the 1/32 condition)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common.config import SEED
from src.data.conditions import Condition


class ThresholdRule(ClassifierMixin, BaseEstimator):
    """
    Predict "edited" from one feature with one cutoff learned on the
    training data: the cutoff and direction (edited below or above it)
    that maximize training balanced accuracy. Ties choose "below" and then
    the lowest cutoff, so the rule is deterministic.
    """

    def fit(self, X, y):
        values = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
        targets = np.asarray(y)
        if not len(values) or not np.isfinite(values).all() or set(np.unique(targets)) != {0, 1}:
            raise ValueError("ThresholdRule needs finite values and both classes")
        self.classes_ = np.array([0, 1])

        # Balanced accuracy of "edited if value <= cutoff" for every candidate
        # cutoff, from cumulative class counts over the sorted values.
        order = np.argsort(values, kind="stable")
        ordered, ordered_targets = values[order], targets[order]
        ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
        edited_below = np.r_[0, np.cumsum(ordered_targets)[ends]]
        human_below = np.r_[0, np.cumsum(1 - ordered_targets)[ends]]
        n_edited, n_human = targets.sum(), len(targets) - targets.sum()
        below = 0.5 * (edited_below / n_edited + (n_human - human_below) / n_human)
        # "Edited if value > cutoff" is the complement, with balanced accuracy 1 - below.
        above = 1.0 - below

        candidates = _cutoff_candidates(ordered)
        if below.max() >= above.max() - 1e-12:
            self.direction_ = "below"
            self.cutoff_ = float(candidates[np.flatnonzero(below >= below.max() - 1e-12)[0]])
        else:
            self.direction_ = "above"
            self.cutoff_ = float(candidates[np.flatnonzero(above >= above.max() - 1e-12)[0]])
        return self

    def predict(self, X):
        values = np.asarray(X, dtype=float).reshape(len(X), -1)[:, 0]
        edited = values <= self.cutoff_ if self.direction_ == "below" else values > self.cutoff_
        return edited.astype(int)


def _cutoff_candidates(values: np.ndarray) -> np.ndarray:
    """
    One cutoff per way of splitting the sorted distinct values: below the
    smallest, between each neighbouring pair, and at the largest.
    """
    unique = np.unique(values)
    midpoints = unique[:-1] / 2 + unique[1:] / 2
    # Adjacent floating-point values have no midpoint strictly between them.
    midpoints = np.where(midpoints >= unique[1:], unique[:-1], midpoints)
    return np.r_[np.nextafter(unique[0], -np.inf), midpoints, unique[-1]]


@dataclass(frozen=True)
class ModelSpec:
    """
    A model: how to build a fresh, unfitted estimator, and which feature
    columns it gets. `inputs` receives the condition and the candidate
    feature columns and returns the columns this model uses.
    """

    name: str
    label: str
    build: Callable[[], object]
    inputs: Callable[[Condition, tuple[str, ...]], tuple[str, ...]] = lambda condition, columns: columns


def _logistic_regression():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)),
    ])


def _random_forest():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED,
        )),
    ])


def _svm():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000, random_state=SEED)),
    ])


def timing_rule_feature(condition: Condition) -> str:
    """The single feature the timing rule uses for this condition."""
    return "mag_timing32_abs_mean" if condition.subdivision == 32 else "mag_timing_abs_mean"


MODELS = (
    ModelSpec("logistic_regression", "Logistic regression", _logistic_regression),
    ModelSpec("random_forest", "Random forest", _random_forest),
    ModelSpec("svm", "SVM (RBF)", _svm),
    ModelSpec(
        "timing_rule", "Timing rule", ThresholdRule,
        inputs=lambda condition, columns: (timing_rule_feature(condition),),
    ),
)
MODELS_BY_NAME = {model.name: model for model in MODELS}
