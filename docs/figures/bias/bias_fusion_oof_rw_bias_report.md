# Bias report: fusion_oof_rw

Urgent recall gap allowed: 0.05. Groups with fewer than 20 urgent images are listed but not used for the gap.

**Result: PASS**

Checked as deployed: global threshold 0.076, per-group thresholds (sex: female 0.048, male 0.074; age_group: 45-60 0.058, 60-75 0.064, 75+ 0.070). Validation rows are scored by thresholds fitted on the other patient half, test rows by thresholds fitted on validation. With one global threshold the result was **FAIL** (see global_threshold/bias_report.md).

## val

### sex: gap 0.012, PASS

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| female | 623 | 97 | 0.907 (0.833–0.950) | 0.172 | 0.289 | 0.307 | 0.820 | yes |
| male | 650 | 112 | 0.920 (0.854–0.957) | 0.215 | 0.349 | 0.409 | 0.735 | yes |
| unknown | 673 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.663 | 0.337 | no |

### age_group: gap 0.038, PASS

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| 0-30 | 82 | 2 | 1.000 (0.342–1.000) | 0.051 | 0.098 | 0.549 | 0.476 | no |
| 30-45 | 225 | 14 | 0.929 (0.685–0.987) | 0.120 | 0.213 | 0.573 | 0.480 | no |
| 45-60 | 588 | 51 | 0.922 (0.815–0.969) | 0.121 | 0.214 | 0.412 | 0.662 | yes |
| 60-75 | 723 | 90 | 0.922 (0.848–0.962) | 0.184 | 0.307 | 0.483 | 0.622 | yes |
| 75+ | 328 | 52 | 0.885 (0.770–0.946) | 0.200 | 0.326 | 0.421 | 0.701 | yes |

### body_site: gap 0.045, PASS

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| acral | 125 | 24 | 0.875 (0.690–0.957) | 0.284 | 0.429 | 0.552 | 0.592 | yes |
| head_neck | 799 | 69 | 0.913 (0.823–0.960) | 0.116 | 0.206 | 0.390 | 0.681 | yes |
| lower_extremity | 66 | 16 | 1.000 (0.806–1.000) | 0.333 | 0.500 | 0.515 | 0.727 | no |
| trunk | 462 | 50 | 0.900 (0.786–0.957) | 0.155 | 0.265 | 0.459 | 0.628 | yes |
| upper_extremity | 494 | 50 | 0.920 (0.812–0.968) | 0.177 | 0.297 | 0.559 | 0.526 | yes |

### skin_tone: gap 0.039, PASS

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| I-II (light) | 879 | 151 | 0.907 (0.850–0.944) | 0.199 | 0.326 | 0.356 | 0.784 | yes |
| III-IV (medium) | 385 | 56 | 0.946 (0.854–0.982) | 0.180 | 0.303 | 0.366 | 0.764 | yes |
| V-VI (dark) | 9 | 2 | 0.500 (0.095–0.905) | 0.167 | 0.250 | 0.333 | 0.667 | no |
| unknown | 673 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.663 | 0.337 | no |

## test

### sex: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| female | 128 | 21 | 1.000 (0.845–1.000) | 0.204 | 0.339 | 0.359 | 0.805 | yes |
| male | 90 | 14 | 0.643 (0.388–0.837) | 0.148 | 0.240 | 0.367 | 0.678 | no |
| unknown | 121 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.661 | 0.339 | no |

### age_group: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| 0-30 | 12 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.583 | 0.417 | no |
| 30-45 | 47 | 1 | 1.000 (0.207–1.000) | 0.059 | 0.111 | 0.660 | 0.362 | no |
| 45-60 | 111 | 8 | 0.750 (0.409–0.929) | 0.097 | 0.171 | 0.477 | 0.559 | no |
| 60-75 | 108 | 10 | 0.700 (0.397–0.892) | 0.099 | 0.173 | 0.380 | 0.657 | no |
| 75+ | 61 | 16 | 1.000 (0.806–1.000) | 0.320 | 0.485 | 0.443 | 0.820 | no |

### body_site: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| acral | 17 | 3 | 1.000 (0.438–1.000) | 0.273 | 0.429 | 0.529 | 0.647 | no |
| head_neck | 133 | 9 | 0.778 (0.453–0.937) | 0.080 | 0.144 | 0.376 | 0.662 | no |
| lower_extremity | 6 | 2 | 1.000 (0.342–1.000) | 0.667 | 0.800 | 0.833 | 0.500 | no |
| trunk | 94 | 10 | 0.800 (0.490–0.943) | 0.145 | 0.246 | 0.479 | 0.585 | no |
| upper_extremity | 89 | 11 | 0.909 (0.623–0.984) | 0.208 | 0.339 | 0.562 | 0.539 | no |

### skin_tone: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| I-II (light) | 149 | 29 | 0.862 (0.694–0.945) | 0.229 | 0.362 | 0.409 | 0.732 | yes |
| III-IV (medium) | 67 | 6 | 0.833 (0.436–0.970) | 0.093 | 0.167 | 0.254 | 0.806 | no |
| V-VI (dark) | 2 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.500 | 0.500 | no |
| unknown | 121 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.661 | 0.339 | no |
