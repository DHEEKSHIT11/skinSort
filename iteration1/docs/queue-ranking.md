# Queue ranking baseline

This module is the first implementation of SkinSort's second-stage queue
component. The image model supplies calibrated probabilities for `routine`,
`priority`, and `urgent`. `QueueRanker` combines those probabilities with wait
time, appointment-window deadlines, and clinician overrides.

The baseline is intentionally transparent while the project collects
clinician-approved ranking examples. It can later be replaced by a learned
ranking model without changing the input/output contract.

## Ranking behavior

- Severity is represented by all three probabilities, not only the argmax label.
- Waiting time grows logarithmically, so old cases gain priority without
  dominating the queue immediately.
- A closing or overdue latest appointment time adds a separate deadline signal.
- Cases above `urgent_threshold` receive a safety override and stay ahead of
  non-overridden cases.
- Future availability windows are marked `not_available` and placed after cases
  that can be scheduled now.
- Clinician overrides are explicit and included in the result for auditability.

```python
from datetime import datetime, timezone

from src.queue.ranker import AppointmentRequest, QueueRanker, SeverityPrediction

request = AppointmentRequest(
    case_id="case-001",
    submitted_at=datetime.now(timezone.utc),
    severity=SeverityPrediction(routine=0.05, priority=0.15, urgent=0.80),
)

result = QueueRanker().rank(request)
print(result.severity_tier, result.score, result.safety_override)
```

The next step for a learned second-stage model is to collect cases with the
severity output, wait-time context, availability constraints, and the final
clinician-approved ordering. Those records can be used to train and evaluate a
learning-to-rank model while retaining the safety checks around it.

## Baseline run

The prototype training entry point is:

```bash
python3 -m scripts.train_baselines /path/to/manifest.jsonl \
  --output reports/baseline_metrics.json \
  --model-output artifacts/severity_model.joblib
```

The current experiment uses 180 public ISIC thumbnails with 60 examples each
for routine, priority, and urgent strata. Model 1 is evaluated on a stratified
25% holdout. Model 2 is evaluated against a transparent prototype scheduling
policy because clinician-approved queue-order labels do not yet exist. These
metrics demonstrate the software pipeline only; they are not clinical
performance or evidence of fairness.

The saved Model 1 artifact exposes `predict_proba` for the three severity
classes and can be reloaded with `joblib.load`. Model 2 consumes those
probabilities rather than a hard severity label.
