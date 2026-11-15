# Bias report: image_main

Urgent recall gap allowed: 0.05. Groups with fewer than 20 urgent images are listed but not used for the gap.

**Result: FAIL** (failed: test/age_group, test/body_site, test/source, fairness/skin_tone)

## test

### sex: gap 0.015, PASS

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| female | 1125 | 172 | 0.860 (0.801–0.904) | 0.509 | 0.639 | 0.852 | 0.259 | yes |
| male | 1675 | 432 | 0.875 (0.840–0.903) | 0.639 | 0.738 | 0.840 | 0.353 | yes |
| unknown | 68 | 26 | 0.923 (0.759–0.979) | 0.615 | 0.738 | 0.750 | 0.574 | no |

### age_group: gap 0.202, FAIL

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| 0-30 | 729 | 21 | 0.714 (0.500–0.862) | 0.312 | 0.435 | 0.947 | 0.066 | yes |
| 30-45 | 505 | 60 | 0.750 (0.628–0.842) | 0.474 | 0.581 | 0.871 | 0.188 | yes |
| 45-60 | 618 | 118 | 0.915 (0.851–0.953) | 0.532 | 0.673 | 0.830 | 0.328 | yes |
| 60-75 | 652 | 247 | 0.862 (0.814–0.900) | 0.653 | 0.743 | 0.775 | 0.500 | yes |
| 75+ | 292 | 155 | 0.916 (0.862–0.950) | 0.679 | 0.780 | 0.726 | 0.716 | yes |
| unknown | 72 | 29 | 0.931 (0.780–0.981) | 0.659 | 0.771 | 0.778 | 0.569 | no |

### body_site: gap 0.259, FAIL

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| acral | 78 | 21 | 0.667 (0.454–0.828) | 0.737 | 0.700 | 0.846 | 0.244 | yes |
| genital | 4 | 0 | n/a (n/a–n/a) | n/a | n/a | 1.000 | 0.000 | no |
| head_neck | 432 | 158 | 0.848 (0.784–0.896) | 0.615 | 0.713 | 0.750 | 0.505 | yes |
| lower_extremity | 415 | 116 | 0.905 (0.838–0.946) | 0.593 | 0.717 | 0.800 | 0.427 | yes |
| trunk | 1008 | 206 | 0.864 (0.811–0.904) | 0.580 | 0.694 | 0.844 | 0.305 | yes |
| unknown | 653 | 21 | 0.905 (0.711–0.973) | 0.358 | 0.514 | 0.945 | 0.081 | no |
| upper_extremity | 278 | 108 | 0.926 (0.861–0.962) | 0.676 | 0.781 | 0.799 | 0.532 | yes |

### skin_tone: gap n/a, TOO_FEW_CASES

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| I-II (light) | 11 | 11 | 0.909 (0.623–0.984) | 1.000 | 0.952 | 0.909 | 0.909 | no |
| III-IV (medium) | 2 | 2 | 1.000 (0.342–1.000) | 1.000 | 1.000 | 1.000 | 1.000 | no |
| unknown | 2855 | 617 | 0.872 (0.843–0.896) | 0.591 | 0.705 | 0.842 | 0.319 | no |

### source: gap 0.127, FAIL

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| ham10000 | 1520 | 152 | 0.776 (0.704–0.835) | 0.327 | 0.460 | 0.818 | 0.237 | yes |
| isic | 1348 | 478 | 0.904 (0.874–0.927) | 0.770 | 0.832 | 0.870 | 0.416 | yes |

## fairness

### skin_tone: gap 0.174, FAIL

| Group | Images | Urgent | Urgent recall (95% CI) | Precision | Urgent F1 | Accuracy (urgent vs not) | Flagged | Used for gap |
|---|---|---|---|---|---|---|---|---|
| I-II (light) | 819 | 129 | 0.605 (0.518–0.685) | 0.199 | 0.299 | 0.554 | 0.479 | yes |
| III-IV (medium) | 2076 | 203 | 0.493 (0.425–0.561) | 0.105 | 0.173 | 0.539 | 0.459 | yes |
| V-VI (dark) | 837 | 66 | 0.667 (0.547–0.768) | 0.103 | 0.179 | 0.517 | 0.509 | yes |
| unknown | 153 | 23 | 0.652 (0.449–0.812) | 0.181 | 0.283 | 0.503 | 0.542 | no |
