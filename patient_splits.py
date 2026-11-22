"""Fail-closed helpers for future patient-disjoint image evaluation.

These functions have synthetic tests and are used in a separate exploratory
LUTSeg pilot. They do not imply that the AZH pilot or WoundsDB metadata audit
used a patient-disjoint model evaluation.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _validate(labels, groups, n_splits):
    labels, groups = np.asarray(labels, dtype=object), np.asarray(groups, dtype=object)
    if labels.ndim != 1 or groups.ndim != 1 or not len(labels) or len(labels) != len(groups):
        raise ValueError("labels and patient groups must be equal nonempty vectors")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if any(not isinstance(x, str) or not x.strip() for x in groups):
        raise ValueError("every image must have a nonempty string patient identifier")
    if any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError("every image must have a nonempty string label")
    labels = np.asarray([x.strip() for x in labels])
    groups = np.asarray([x.strip() for x in groups])
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("at least two classes are required")
    group_counts = {str(label): len(set(groups[labels == label])) for label in classes}
    too_rare = {label: count for label, count in group_counts.items() if count < n_splits}
    if too_rare:
        raise ValueError(f"Fewer than {n_splits} patient groups in one or more classes: {too_rare}")
    return labels, groups, set(classes)


def patient_disjoint_folds(labels, groups, n_splits=5, seed=20261121):
    """Return deterministic outer folds and reject incomplete group stratification."""
    labels, groups, classes = _validate(labels, groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    seen = []
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("patient group appears in both training and test")
        if set(labels[train]) != classes or set(labels[test]) != classes:
            raise ValueError("The generated split omits a class; more independent groups or a revised design is needed")
        seen.extend(test.tolist())
    if sorted(seen) != list(range(len(labels))):
        raise AssertionError("each image must appear in exactly one held-out fold")
    return folds


def patient_disjoint_inner(labels, groups, outer_train, n_splits=5, seed=20261122):
    """Split outer-train patients into model, calibration and selector subsets.

    Uses two validation folds for calibration and selector fitting, and the
    remaining folds for model fitting. Returns indices in the original arrays.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    outer_train = np.asarray(outer_train)
    if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
        raise ValueError("labels and patient groups must be equal-length vectors")
    if outer_train.ndim != 1 or not len(outer_train) or len(set(outer_train.tolist())) != len(outer_train):
        raise ValueError("outer_train must be a nonempty vector of unique indices")
    if not np.issubdtype(outer_train.dtype, np.integer) or np.any(outer_train < 0) or np.any(outer_train >= len(labels)):
        raise ValueError("outer_train indices must be in range")
    if n_splits < 3:
        raise ValueError("at least three inner folds are required")
    folds = patient_disjoint_folds(labels[outer_train], groups[outer_train], n_splits, seed)
    cal = outer_train[folds[0][1]]
    selector = outer_train[folds[1][1]]
    model = outer_train[np.concatenate([fold[1] for fold in folds[2:]])]
    parts = (model, cal, selector)
    group_sets = [set(groups[part]) for part in parts]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("patient groups overlap within the training pool")
    if set(np.concatenate(parts)) != set(outer_train):
        raise AssertionError("inner partitions must cover the outer training pool")
    return parts
