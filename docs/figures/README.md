# Figures

Copied from `outputs/` by `python -m src.training.collect_figures`. Discussed in [../model_pipeline_report.md](../model_pipeline_report.md).

## selection

| File | What it shows | Source |
|---|---|---|
| [selection/selection_backbone_comparison.png](selection/selection_backbone_comparison.png) | Backbones: validation urgent AUC, size and CPU speed | `outputs/selection/backbone_comparison.png` |
| [selection/selection_phone_comparison.png](selection/selection_phone_comparison.png) | Phone candidates and baselines (selection metric) | `outputs/selection/phone_comparison.png` |
| [selection/selection_model_selection.md](selection/selection_model_selection.md) | Selection tables and the reason for the choice | `outputs/selection/model_selection.md` |

## validation

| File | What it shows | Source |
|---|---|---|
| [validation/evaluation_image_main_confusion_matrix.png](validation/evaluation_image_main_confusion_matrix.png) | Image model, main test set | `outputs/evaluation/image_main/confusion_matrix.png` |
| [validation/evaluation_image_main_calibration.png](validation/evaluation_image_main_calibration.png) | Image model calibration, main test set | `outputs/evaluation/image_main/calibration.png` |
| [validation/evaluation_fusion_oof_rw_confusion_matrix.png](validation/evaluation_fusion_oof_rw_confusion_matrix.png) | Selected phone model, pad_test | `outputs/evaluation/fusion_oof_rw/confusion_matrix.png` |
| [validation/evaluation_fusion_oof_rw_calibration.png](validation/evaluation_fusion_oof_rw_calibration.png) | Phone model calibration, pad_test | `outputs/evaluation/fusion_oof_rw/calibration.png` |

## bias

| File | What it shows | Source |
|---|---|---|
| [bias/bias_fusion_oof_rw_val_sex.png](bias/bias_fusion_oof_rw_val_sex.png) | Phone model, urgent recall by sex (out-of-fold) | `outputs/bias/fusion_oof_rw/val_sex.png` |
| [bias/bias_fusion_oof_rw_val_body_site.png](bias/bias_fusion_oof_rw_val_body_site.png) | Phone model, by body site | `outputs/bias/fusion_oof_rw/val_body_site.png` |
| [bias/bias_fusion_oof_rw_val_skin_tone.png](bias/bias_fusion_oof_rw_val_skin_tone.png) | Phone model, by skin tone | `outputs/bias/fusion_oof_rw/val_skin_tone.png` |
| [bias/bias_fusion_oof_rw_val_age_group.png](bias/bias_fusion_oof_rw_val_age_group.png) | Phone model, by age group | `outputs/bias/fusion_oof_rw/val_age_group.png` |
| [bias/bias_fusion_oof_rw_bias_report.md](bias/bias_fusion_oof_rw_bias_report.md) | Phone model bias report | `outputs/bias/fusion_oof_rw/bias_report.md` |
| [bias/bias_fusion_oof_rw_mitigation.md](bias/bias_fusion_oof_rw_mitigation.md) | Phone model mitigation (per-group thresholds) | `outputs/bias/fusion_oof_rw/mitigation.md` |
| [bias/bias_image_main_test_age_group.png](bias/bias_image_main_test_age_group.png) | Image model, by age group (test) | `outputs/bias/image_main/test_age_group.png` |
| [bias/bias_image_main_test_body_site.png](bias/bias_image_main_test_body_site.png) | Image model, by body site (test) | `outputs/bias/image_main/test_body_site.png` |
| [bias/bias_image_main_test_source.png](bias/bias_image_main_test_source.png) | Image model, by source (test) | `outputs/bias/image_main/test_source.png` |
| [bias/bias_image_main_fairness_skin_tone.png](bias/bias_image_main_fairness_skin_tone.png) | Image model, by skin tone (fairness_test) | `outputs/bias/image_main/fairness_skin_tone.png` |
| [bias/bias_image_main_bias_report.md](bias/bias_image_main_bias_report.md) | Image model bias report | `outputs/bias/image_main/bias_report.md` |
| [bias/bias_image_main_mitigation.md](bias/bias_image_main_mitigation.md) | Image model mitigation | `outputs/bias/image_main/mitigation.md` |

## tuning

| File | What it shows | Source |
|---|---|---|
| [tuning/tuning_b3_sensitivity.png](tuning/tuning_b3_sensitivity.png) | Hyperopt trials on the final backbone: CV AUC vs each hyperparameter | `outputs/tuning_b3/sensitivity.png` |
| [tuning/tuning_b3_search_progress.png](tuning/tuning_b3_search_progress.png) | Hyperopt search progress (final backbone) | `outputs/tuning_b3/search_progress.png` |
| [tuning/tuning_b3_lr_sweep.png](tuning/tuning_b3_lr_sweep.png) | Learning rate, one at a time | `outputs/tuning_b3/lr_sweep.png` |
| [tuning/tuning_sensitivity.png](tuning/tuning_sensitivity.png) | First search, on the leaky fine-tuned features (kept for the write-up) | `outputs/tuning/sensitivity.png` |

## sensitivity

| File | What it shows | Source |
|---|---|---|
| [sensitivity/explain_fusion_oof_rw_shap_importance.png](sensitivity/explain_fusion_oof_rw_shap_importance.png) | SHAP: what moves P(urgent) | `outputs/explain/fusion_oof_rw/shap_importance.png` |
| [sensitivity/explain_fusion_oof_rw_shap_answers.png](sensitivity/explain_fusion_oof_rw_shap_answers.png) | SHAP per answer (yes vs no) | `outputs/explain/fusion_oof_rw/shap_answers.png` |
| [sensitivity/analysis_robustness.png](sensitivity/analysis_robustness.png) | Robustness to blur, light, compression, resolution | `outputs/analysis/robustness.png` |
| [sensitivity/analysis_thresholds.png](sensitivity/analysis_thresholds.png) | Recall vs flagged share for every threshold | `outputs/analysis/thresholds.png` |
