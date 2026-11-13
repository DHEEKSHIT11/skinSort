"""Small, reproducible image-severity baseline for the SkinSort prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEVERITY_LABELS = ("routine", "priority", "urgent")


def image_features(path: str | Path) -> np.ndarray:
    """Extract lightweight visual features without a deep-learning runtime."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0

    gray = array.mean(axis=2)
    vertical_edges = np.diff(gray, axis=0)
    horizontal_edges = np.diff(gray, axis=1)
    features = [*array.mean(axis=(0, 1)), *array.std(axis=(0, 1))]
    features.extend([float(gray.mean()), float(gray.std())])
    features.extend([
        float(np.abs(vertical_edges).mean()),
        float(np.abs(horizontal_edges).mean()),
        float((vertical_edges ** 2).mean()),
        float((horizontal_edges ** 2).mean()),
    ])
    features.extend(np.percentile(array.reshape(-1, 3), [10, 25, 50, 75, 90], axis=0).ravel())
    features.extend(array.reshape(2, 32, 2, 32, 3).mean(axis=(1, 3)).ravel())
    for channel in range(3):
        features.extend(np.histogram(array[:, :, channel], bins=8, range=(0, 1), density=True)[0])
    return np.asarray(features, dtype=np.float32)


def load_manifest(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load JSONL records produced by the ISIC sample downloader."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    features = np.vstack([image_features(row["path"]) for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    identifiers = [row["image_id"] for row in rows]
    return features, labels, identifiers


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    test_size: float = 0.25,
) -> dict[str, Any]:
    """Train and evaluate a stratified holdout severity classifier."""
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    metrics = {
        "train_rows": int(len(train_y)),
        "test_rows": int(len(test_y)),
        "labels": list(model.classes_),
        "accuracy": float(accuracy_score(test_y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "macro_f1": float(f1_score(test_y, predictions, average="macro")),
        "log_loss": float(log_loss(test_y, probabilities, labels=model.classes_)),
        "confusion_matrix": confusion_matrix(test_y, predictions, labels=model.classes_).tolist(),
        "recall_by_class": {
            label: float(recall_score(test_y, predictions, labels=[label], average="macro"))
            for label in model.classes_
        },
        "class_counts_test": {
            label: int((test_y == label).sum()) for label in SEVERITY_LABELS
        },
    }
    return {
        "model": model,
        "metrics": metrics,
        "test_features": test_x,
        "test_labels": test_y,
        "test_probabilities": probabilities,
    }


def probability_rows(model_result: dict[str, Any]) -> list[dict[str, float]]:
    """Convert sklearn probabilities to the Model 1 contract."""
    classes = list(model_result["model"].classes_)
    rows = []
    for values in model_result["test_probabilities"]:
        row = {label: 0.0 for label in SEVERITY_LABELS}
        row.update({label: float(value) for label, value in zip(classes, values)})
        rows.append(row)
    return rows
