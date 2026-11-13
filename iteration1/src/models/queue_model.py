"""Learned second-stage queue model for prototype evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def queue_features(
    probabilities: Iterable[dict[str, float]],
    wait_hours: Iterable[float],
    hours_to_latest: Iterable[float | None],
) -> np.ndarray:
    """Build Model 2 features from Model 1 probabilities and queue context."""
    rows = []
    for probability, wait, latest in zip(probabilities, wait_hours, hours_to_latest):
        latest_value = 0.0 if latest is None else float(latest)
        rows.append([
            float(probability["routine"]),
            float(probability["priority"]),
            float(probability["urgent"]),
            float(wait),
            latest_value,
            float(latest is None),
            float(latest is not None and latest <= 0),
        ])
    return np.asarray(rows, dtype=np.float64)


def policy_score(features: np.ndarray, urgent_threshold: float = 0.70) -> np.ndarray:
    """Create transparent prototype targets for ranking-model development.

    These targets are an operational policy, not clinician labels or medical
    truth. They make it possible to test the second-stage plumbing before real
    approved ranking labels exist.
    """
    routine, priority, urgent, wait, latest, missing_deadline, overdue = features.T
    severity = 10 * routine + 50 * priority + 100 * urgent
    wait_component = 8 * np.log1p(np.maximum(wait, 0))
    deadline_fraction = np.where(
        missing_deadline == 1,
        0,
        np.where(overdue == 1, 1, np.maximum(0, 1 - latest / 72)),
    )
    deadline_component = 25 * deadline_fraction
    safety_component = np.where(urgent >= urgent_threshold, 10000, 0)
    return severity + wait_component + deadline_component + safety_component


def train_queue_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    seed: int = 42,
) -> HistGradientBoostingRegressor:
    """Train a nonlinear ranker on approved or prototype policy scores."""
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(train_features, train_targets)
    return model


def safety_adjusted_scores(
    features: np.ndarray,
    predicted_scores: np.ndarray,
    urgent_threshold: float = 0.70,
) -> np.ndarray:
    """Apply the non-negotiable urgent-case guard after learned scoring."""
    if not 0 < urgent_threshold <= 1:
        raise ValueError("urgent_threshold must be between 0 and 1")
    urgent_override = features[:, 2] >= urgent_threshold
    return predicted_scores + np.where(urgent_override, 10000.0, 0.0)


def pairwise_accuracy(predicted: np.ndarray, target: np.ndarray, seed: int = 42) -> float:
    """Measure how often the model orders a random pair like the target policy."""
    if len(predicted) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(predicted), size=10000)
    right = rng.integers(0, len(predicted), size=10000)
    valid = left != right
    left, right = left[valid], right[valid]
    target_delta = target[left] - target[right]
    predicted_delta = predicted[left] - predicted[right]
    target_sign = np.sign(target_delta)
    predicted_sign = np.sign(predicted_delta)
    return float(np.mean((target_sign == predicted_sign) | (target_sign == 0)))


def rank_correlation(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return Spearman-style rank correlation without requiring scipy."""
    predicted_ranks = np.argsort(np.argsort(predicted)).astype(float)
    target_ranks = np.argsort(np.argsort(target)).astype(float)
    if len(predicted) < 2:
        return 1.0
    return float(np.corrcoef(predicted_ranks, target_ranks)[0, 1])
