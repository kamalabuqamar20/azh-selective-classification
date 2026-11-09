"""Validation and evaluation helpers for the exploratory AZH analysis."""
from __future__ import annotations

from collections.abc import Iterator
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_azh_pilot import CLASSES, risk_summary


def validated_risk(errors: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                   coverage: float) -> dict:
    """Evaluate a score at equal image coverage; labels are used only in audit."""
    errors = np.asarray(errors)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if errors.ndim != 1 or labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("errors, labels and scores must be one-dimensional")
    if not len(errors) or len(errors) != len(labels) or len(labels) != len(scores):
        raise ValueError("errors, labels and scores must have the same nonzero length")
    if not np.isin(errors, [0, 1]).all():
        raise ValueError("errors must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores must all be finite")
    if not np.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    if set(labels.tolist()) != set(CLASSES):
        raise ValueError("every configured wound class must be present")
    return risk_summary(errors.astype(int), labels, scores.astype(float), float(coverage))


def outer_folds(labels: np.ndarray, n_splits: int = 5,
                seed: int = 20261109) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stratified image-level folds; these are not patient-grouped."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or set(labels.tolist()) != set(CLASSES):
        raise ValueError("outer-fold labels must be a nonempty array of all four classes")
    _, counts = np.unique(labels, return_counts=True)
    if not isinstance(n_splits, int) or n_splits < 2 or np.min(counts) < n_splits:
        raise ValueError("each class must contain at least n_splits images")
    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(labels))
    yield from split.split(indices, labels)


def inner_partitions(labels: np.ndarray, outer_train: np.ndarray,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an outer-training fold into model, temperature and selector sets."""
    labels, outer_train = np.asarray(labels), np.asarray(outer_train)
    fit, validation = train_test_split(
        outer_train, test_size=.30, stratify=labels[outer_train], random_state=seed
    )
    calibration, selector = train_test_split(
        validation, test_size=.50, stratify=labels[validation], random_state=seed
    )
    sets = [set(x.tolist()) for x in (fit, calibration, selector)]
    if sum(map(len, sets)) != len(set.union(*sets)) or set.union(*sets) != set(outer_train):
        raise AssertionError("inner partitions must form a disjoint cover of outer train")
    return fit, calibration, selector


def selector_inputs(probabilities: np.ndarray, focus: np.ndarray) -> np.ndarray:
    """Features for correctness models; true wound class is never an input."""
    probabilities, focus = np.asarray(probabilities), np.asarray(focus)
    if probabilities.ndim != 2 or focus.ndim != 1 or len(probabilities) != len(focus):
        raise ValueError("probabilities must be two-dimensional and match focus")
    if not np.isfinite(probabilities).all() or not np.isfinite(focus).all():
        raise ValueError("selector inputs must be finite")
    if np.any(focus < 0) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("invalid probability or focus values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7):
        raise ValueError("class probabilities must sum to one")
    confidence = logit(np.clip(probabilities.max(axis=1), 1e-8, 1 - 1e-8))
    return np.column_stack([confidence, np.log1p(focus)])


def fit_correctness_selectors(probabilities: np.ndarray, focus: np.ndarray,
                              correct: np.ndarray):
    """Fit like-for-like one- and two-feature selectors on the same samples."""
    features = selector_inputs(probabilities, focus)
    correct = np.asarray(correct)
    if correct.ndim != 1 or len(correct) != len(features) or set(correct.tolist()) != {0, 1}:
        raise ValueError("selector-fit data must include correct and incorrect predictions")
    one = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, solver="lbfgs"))
    two = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, solver="lbfgs"))
    one.fit(features[:, :1], correct)
    two.fit(features, correct)
    return one, two


def scores_for_policies(probabilities: np.ndarray, focus: np.ndarray, one, two) -> dict[str, np.ndarray]:
    """Compute scores without access to the ground-truth wound type."""
    features = selector_inputs(probabilities, focus)
    return {
        "confidence": np.asarray(probabilities).max(axis=1),
        "focus": np.asarray(focus),
        "learned_confidence": one.predict_proba(features[:, :1])[:, 1],
        "confidence_plus_focus": two.predict_proba(features)[:, 1],
    }
