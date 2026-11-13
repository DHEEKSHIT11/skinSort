"""Train Model 1 and prototype Model 2 on a local manifest.

The severity metrics use a held-out ISIC sample. The queue metrics evaluate
Model 2 against an explicit prototype scheduling policy because clinician-
approved ranking labels are not yet available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.models.queue_model import (
    pairwise_accuracy,
    policy_score,
    queue_features,
    rank_correlation,
    safety_adjusted_scores,
    train_queue_model,
)
from src.models.severity_baseline import load_manifest, probability_rows, train_model


def run(
    manifest: Path,
    output: Path,
    seed: int = 42,
    model_output: Path | None = None,
) -> dict:
    features, labels, _ = load_manifest(manifest)
    severity_result = train_model(features, labels, seed=seed)
    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(severity_result["model"], model_output)
    probabilities = probability_rows(severity_result)

    rng = np.random.default_rng(seed)
    train_probabilities = [{
        "routine": float(row[0]),
        "priority": float(row[1]),
        "urgent": float(row[2]),
    } for row in rng.dirichlet([2.5, 1.5, 1.0], size=5000)]
    train_wait = rng.uniform(0, 24 * 30, size=len(train_probabilities))
    train_latest = rng.uniform(-24, 24 * 30, size=len(train_probabilities))
    train_features = queue_features(train_probabilities, train_wait, train_latest)
    train_targets = policy_score(train_features)
    queue_model = train_queue_model(train_features, train_targets, seed=seed)

    test_wait = rng.uniform(0, 24 * 30, size=len(probabilities))
    test_latest = rng.uniform(-24, 24 * 30, size=len(probabilities))
    test_features = queue_features(probabilities, test_wait, test_latest)
    test_targets = policy_score(test_features)
    raw_predictions = queue_model.predict(test_features)
    test_predictions = safety_adjusted_scores(test_features, raw_predictions)
    if not all(np.isclose(sum(row.values()), 1.0, atol=1e-6) for row in probabilities):
        raise ValueError("Model 1 probabilities failed the Model 2 input contract")
    queue_metrics = {
        "train_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "pairwise_accuracy_vs_prototype_policy": pairwise_accuracy(
            test_predictions, test_targets, seed=seed
        ),
        "rank_correlation_vs_prototype_policy": rank_correlation(
            test_predictions, test_targets
        ),
    }

    report = {
        "data": {
            "manifest": str(manifest),
            "rows": int(len(labels)),
            "label_counts": {label: int((labels == label).sum()) for label in sorted(set(labels))},
            "source": "ISIC public thumbnail sample",
        },
        "model_1_severity": severity_result["metrics"],
        "model_2_queue": {
            "metrics": queue_metrics,
            "target_definition": "transparent prototype policy, not clinician labels",
            "features": [
                "p_routine", "p_priority", "p_urgent", "wait_hours",
                "hours_to_latest", "missing_deadline", "overdue",
            ],
        },
        "merged_pipeline": {
            "model_1_probabilities_passed_to_model_2": True,
            "model_1_probability_rows": int(len(probabilities)),
            "model_2_queue_scores": int(len(test_predictions)),
            "urgent_safety_overrides": int((test_features[:, 2] >= 0.70).sum()),
        },
    }
    if model_output is not None:
        report["model_1_severity"]["artifact"] = str(model_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.manifest, args.output, seed=args.seed, model_output=args.model_output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
