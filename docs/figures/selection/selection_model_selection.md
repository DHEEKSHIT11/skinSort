# Model selection

Selection uses validation data only. Test columns are shown for the report and never used to choose.

## Stage 1: image backbone (`mlflow.search_runs(order_by=['metrics.best_urgent_auc DESC'])`)

| Backbone | Val urgent AUC | Val urgent recall | Val macro F1 | Params (M) | Checkpoint MB | CPU ms/photo |
|---|---|---|---|---|---|---|
| efficientnet_b3 ✅ | 0.932 | 0.812 | 0.776 | 10.7 | 43 | 199 |
| resnet50 | 0.929 | 0.895 | 0.754 | 23.5 | 94 | 34 |
| convnext_tiny | 0.877 | 0.851 | 0.626 | 27.8 | 111 | 503 |

Selected: **efficientnet_b3** (highest validation urgent AUC).

## Stage 2: phone triage model (`order_by=['metrics.val_urgent_auc DESC']`)

Filters: validation calibration error ≤ 0.05; tau fitted on ≥ 100 urgent photos; servable as a registry model. Bias: among candidates within 0.02 AUC of the best, the one with the fewest failing bias slices wins.

| Candidate | Kind | Val AUC | Val calibration | Tau from (urgent) | Bias (failing slices) | Eligible | Test AUC | Test recall |
|---|---|---|---|---|---|---|---|---|
| fusion_pad_tuned | fusion | 0.869 | 0.020 | pad_val (31) | INCONCLUSIVE (0) | no | 0.730 | 0.543 |
| fusion_oof | fusion | 0.813 | 0.032 | out-of-fold pad_train + pad_val (209) | FAIL (3) | yes | 0.707 | 0.743 |
| fusion_oof_rw ✅ | fusion | 0.807 | 0.031 | out-of-fold pad_train + pad_val (209) | FAIL (1) | yes | 0.694 | 0.743 |
| fusion_pad | fusion | 0.806 | 0.015 | pad_val (31) | INCONCLUSIVE (0) | no | 0.717 | 0.829 |
| fusion_oof_tuned | fusion | 0.800 | 0.037 | out-of-fold pad_train + pad_val (209) | FAIL (3) | yes | 0.677 | 0.800 |
| baseline_answers_lr | baseline | 0.781 | 0.034 | out-of-fold pad_train + pad_val (209) | FAIL (3) | no | 0.577 | 0.486 |
| baseline_image_only | baseline | 0.663 | 0.018 | out-of-fold pad_train + pad_val (209) | FAIL (1) | no | 0.645 | 0.829 |
| baseline_prevalence | baseline | 0.500 | 0.018 | out-of-fold pad_train + pad_val (209) | PASS (0) | no | 0.500 | 1.000 |

Selected: **fusion_oof_rw** — fewest failing bias slices (1 vs 3 for fusion_oof) within 0.02 validation AUC of the best.
