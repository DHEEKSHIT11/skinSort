# Mitigation: per-group thresholds

Each group gets its own urgent threshold (never higher than the global one), fitted so the group reaches the recall target on the fitting rows.

| Split | Slice | Fitted on | Gap before → after | Worst group before → after | Urgent recall before → after | Flagged before → after | Status after |
|---|---|---|---|---|---|---|---|
| val | sex | val, cross-fitted by patient halves | 0.069 → 0.039 | female → female | 0.852 → 0.866 | 51.6% → 55.2% | PASS |
| test | sex | val | n/a → n/a | None → None | 0.743 → 0.771 | 48.7% → 50.7% | TOO_FEW_CASES |
| val | body_site | val, cross-fitted by patient halves | 0.050 → 0.036 | trunk → acral | 0.852 → 0.856 | 51.6% → 52.0% | PASS |
| test | body_site | val | n/a → n/a | None → None | 0.743 → 0.743 | 48.7% → 49.9% | TOO_FEW_CASES |
| val | skin_tone | val, cross-fitted by patient halves | 0.022 → 0.014 | III-IV (medium) → I-II (light) | 0.852 → 0.861 | 51.6% → 52.3% | PASS |
| test | skin_tone | val | n/a → n/a | None → None | 0.743 → 0.743 | 48.7% → 49.6% | TOO_FEW_CASES |

Group thresholds:

- val / sex (global 0.076): female 0.062, male 0.076
- test / sex (global 0.076): female 0.061, male 0.076
- val / body_site (global 0.076): head_neck 0.076, trunk 0.074, upper_extremity 0.075
- test / body_site (global 0.076): acral 0.058, head_neck 0.076, trunk 0.074, upper_extremity 0.076
- val / skin_tone (global 0.076): I-II (light) 0.075, III-IV (medium) 0.073
- test / skin_tone (global 0.076): I-II (light) 0.076, III-IV (medium) 0.074

## Per group, before → after

### val / sex

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| female | 97 | 0.814 → 0.845 | 62.8% → 74.2% |
| male | 112 | 0.884 → 0.884 | 66.6% → 66.6% |
| unknown | 0 | n/a → n/a | 26.7% → 26.7% |

### test / sex

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| female | 21 | 0.905 → 0.952 | 66.4% → 71.9% |
| male | 14 | 0.500 → 0.500 | 55.6% → 55.6% |
| unknown | 0 | n/a → n/a | 24.8% → 24.8% |

### val / body_site

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| acral | 24 | 0.833 → 0.833 | 49.6% → 49.6% |
| head_neck | 69 | 0.870 → 0.870 | 55.4% → 55.4% |
| lower_extremity | 16 | 0.875 → 0.875 | 65.2% → 65.2% |
| trunk | 50 | 0.820 → 0.840 | 52.6% → 53.9% |
| upper_extremity | 50 | 0.860 → 0.860 | 43.1% → 43.3% |

### test / body_site

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| acral | 3 | 1.000 → 1.000 | 52.9% → 64.7% |
| head_neck | 9 | 0.556 → 0.556 | 54.9% → 54.9% |
| lower_extremity | 2 | 1.000 → 1.000 | 50.0% → 50.0% |
| trunk | 10 | 0.600 → 0.600 | 42.6% → 44.7% |
| upper_extremity | 11 | 0.909 → 0.909 | 44.9% → 44.9% |

### val / skin_tone

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| I-II (light) | 151 | 0.861 → 0.861 | 65.6% → 65.9% |
| III-IV (medium) | 56 | 0.839 → 0.875 | 62.9% → 65.7% |
| V-VI (dark) | 2 | 0.500 → 0.500 | 55.6% → 55.6% |
| unknown | 0 | n/a → n/a | 26.7% → 26.7% |

### test / skin_tone

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| I-II (light) | 29 | 0.759 → 0.759 | 62.4% → 62.4% |
| III-IV (medium) | 6 | 0.667 → 0.667 | 61.2% → 65.7% |
| V-VI (dark) | 0 | n/a → n/a | 50.0% → 50.0% |
| unknown | 0 | n/a → n/a | 24.8% → 24.8% |

