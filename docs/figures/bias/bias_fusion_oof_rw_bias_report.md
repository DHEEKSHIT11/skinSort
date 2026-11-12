# Bias report: fusion_oof_rw

Urgent recall gap allowed: 0.05. Groups with fewer than 20 urgent images are listed but not used for the gap.

**Result: FAIL** (failed: val/sex)

## val

### sex: gap 0.069, FAIL

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| female | 623 | 97 | 0.814 (0.726–0.879) | 0.202 | 0.324 | 0.470 | 0.628 | yes |
| male | 650 | 112 | 0.884 (0.811–0.931) | 0.229 | 0.363 | 0.466 | 0.666 | yes |
| unknown | 673 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.733 | 0.267 | no |

### age_group: gap 0.043, PASS

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| 0-30 | 82 | 2 | 1.000 (0.342–1.000) | 0.057 | 0.108 | 0.598 | 0.427 | no |
| 30-45 | 225 | 14 | 0.929 (0.685–0.987) | 0.124 | 0.218 | 0.587 | 0.467 | no |
| 45-60 | 588 | 51 | 0.824 (0.697–0.904) | 0.141 | 0.241 | 0.551 | 0.505 | yes |
| 60-75 | 723 | 90 | 0.867 (0.781–0.922) | 0.203 | 0.329 | 0.560 | 0.531 | yes |
| 75+ | 328 | 52 | 0.827 (0.703–0.906) | 0.235 | 0.366 | 0.546 | 0.558 | yes |

### body_site: gap 0.050, PASS

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| acral | 125 | 24 | 0.833 (0.641–0.933) | 0.323 | 0.465 | 0.632 | 0.496 | yes |
| head_neck | 799 | 69 | 0.870 (0.770–0.930) | 0.135 | 0.234 | 0.509 | 0.554 | yes |
| lower_extremity | 66 | 16 | 0.875 (0.640–0.965) | 0.326 | 0.475 | 0.530 | 0.652 | no |
| trunk | 462 | 50 | 0.820 (0.692–0.902) | 0.169 | 0.280 | 0.543 | 0.526 | yes |
| upper_extremity | 494 | 50 | 0.860 (0.738–0.930) | 0.202 | 0.327 | 0.642 | 0.431 | yes |

### skin_tone: gap 0.022, PASS

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| I-II (light) | 879 | 151 | 0.861 (0.797–0.907) | 0.225 | 0.357 | 0.468 | 0.656 | yes |
| III-IV (medium) | 385 | 56 | 0.839 (0.722–0.913) | 0.194 | 0.315 | 0.470 | 0.629 | yes |
| V-VI (dark) | 9 | 2 | 0.500 (0.095–0.905) | 0.200 | 0.286 | 0.444 | 0.556 | no |
| unknown | 673 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.733 | 0.267 | no |

## test

### sex: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| female | 128 | 21 | 0.905 (0.711–0.973) | 0.224 | 0.358 | 0.469 | 0.664 | yes |
| male | 90 | 14 | 0.500 (0.268–0.732) | 0.140 | 0.219 | 0.444 | 0.556 | no |
| unknown | 121 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.752 | 0.248 | no |

### age_group: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| 0-30 | 12 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.667 | 0.333 | no |
| 30-45 | 47 | 1 | 1.000 (0.207–1.000) | 0.071 | 0.133 | 0.723 | 0.298 | no |
| 45-60 | 111 | 8 | 0.625 (0.306–0.863) | 0.106 | 0.182 | 0.595 | 0.423 | no |
| 60-75 | 108 | 10 | 0.500 (0.237–0.763) | 0.089 | 0.152 | 0.481 | 0.519 | no |
| 75+ | 61 | 16 | 0.938 (0.717–0.989) | 0.341 | 0.500 | 0.508 | 0.721 | no |

### body_site: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| acral | 17 | 3 | 1.000 (0.438–1.000) | 0.333 | 0.500 | 0.647 | 0.529 | no |
| head_neck | 133 | 9 | 0.556 (0.267–0.811) | 0.068 | 0.122 | 0.459 | 0.549 | no |
| lower_extremity | 6 | 2 | 1.000 (0.342–1.000) | 0.667 | 0.800 | 0.833 | 0.500 | no |
| trunk | 94 | 10 | 0.600 (0.313–0.832) | 0.150 | 0.240 | 0.596 | 0.426 | no |
| upper_extremity | 89 | 11 | 0.909 (0.623–0.984) | 0.250 | 0.392 | 0.652 | 0.449 | no |

### skin_tone: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| I-II (light) | 149 | 29 | 0.759 (0.579–0.878) | 0.237 | 0.361 | 0.477 | 0.624 | yes |
| III-IV (medium) | 67 | 6 | 0.667 (0.300–0.903) | 0.098 | 0.170 | 0.418 | 0.612 | no |
| V-VI (dark) | 2 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.500 | 0.500 | no |
| unknown | 121 | 0 | n/a (n/a–n/a) | 0.000 | 0.000 | 0.752 | 0.248 | no |
