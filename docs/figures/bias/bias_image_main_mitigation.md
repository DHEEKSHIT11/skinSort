# Mitigation: per-group thresholds

Each group gets its own urgent threshold (never higher than the global one), fitted so the group reaches the recall target on the fitting rows.

| Split | Slice | Fitted on | Gap before → after | Worst group before → after | Urgent recall before → after | Flagged before → after | Status after |
|---|---|---|---|---|---|---|---|
| test | age_group | val | 0.202 → 0.209 | 0-30 → 0-30 | 0.873 → 0.879 | 32.1% → 33.5% | FAIL |
| test | body_site | val | 0.259 → 0.259 | acral → acral | 0.873 → 0.887 | 32.1% → 33.9% | FAIL |
| test | source | val | 0.127 → 0.002 | ham10000 → ham10000 | 0.873 → 0.903 | 32.1% → 36.6% | PASS |
| test | skin_tone | val | n/a → n/a | None → None | 0.873 → 0.873 | 32.1% → 32.1% | TOO_FEW_CASES |
| fairness | skin_tone | fairness, cross-fitted by patient halves | 0.174 → 0.043 | III-IV (medium) → V-VI (dark) | 0.563 → 0.867 | 47.7% → 84.7% | PASS |

Group thresholds:

- test / age_group (global 0.173): 30-45 0.122, 45-60 0.137, 60-75 0.173, 75+ 0.173
- test / body_site (global 0.173): head_neck 0.173, lower_extremity 0.173, trunk 0.119, upper_extremity 0.173
- test / source (global 0.173): ham10000 0.098, isic 0.173
- test / skin_tone (global 0.173): none (too few cases)
- fairness / skin_tone (global 0.173): I-II (light) 0.070, III-IV (medium) 0.060, V-VI (dark) 0.076

## Per group, before → after

### test / age_group

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| 0-30 | 21 | 0.714 → 0.714 | 6.6% → 6.6% |
| 30-45 | 60 | 0.750 → 0.800 | 18.8% → 23.4% |
| 45-60 | 118 | 0.915 → 0.924 | 32.8% → 35.3% |
| 60-75 | 247 | 0.862 → 0.862 | 50.0% → 50.0% |
| 75+ | 155 | 0.916 → 0.916 | 71.6% → 71.6% |
| unknown | 29 | 0.931 → 0.931 | 56.9% → 56.9% |

### test / body_site

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| acral | 21 | 0.667 → 0.667 | 24.4% → 24.4% |
| genital | 0 | n/a → n/a | 0.0% → 0.0% |
| head_neck | 158 | 0.848 → 0.848 | 50.5% → 50.5% |
| lower_extremity | 116 | 0.905 → 0.905 | 42.7% → 42.7% |
| trunk | 206 | 0.864 → 0.908 | 30.5% → 35.5% |
| unknown | 21 | 0.905 → 0.905 | 8.1% → 8.1% |
| upper_extremity | 108 | 0.926 → 0.926 | 53.2% → 53.2% |

### test / source

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| ham10000 | 152 | 0.776 → 0.901 | 23.8% → 32.2% |
| isic | 478 | 0.904 → 0.904 | 41.6% → 41.6% |

### test / skin_tone

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| I-II (light) | 11 | 0.909 → 0.909 | 90.9% → 90.9% |
| III-IV (medium) | 2 | 1.000 → 1.000 | 100.0% → 100.0% |
| unknown | 617 | 0.872 → 0.872 | 31.9% → 31.9% |

### fairness / skin_tone

| Group | Urgent | Recall before → after | Flagged before → after |
|---|---|---|---|
| I-II (light) | 129 | 0.605 → 0.876 | 47.9% → 84.4% |
| III-IV (medium) | 203 | 0.493 → 0.892 | 45.9% → 88.4% |
| V-VI (dark) | 66 | 0.667 → 0.848 | 50.9% → 81.4% |
| unknown | 23 | 0.652 → 0.652 | 54.2% → 54.2% |

