# Model Pipeline — requirements, status and working log

> **The final write-up is [model_pipeline_report.md](model_pipeline_report.md).** This file is the working log: sections 2–3 record results in the order they happened, so early entries describe states that were later fixed.

Source of requirements: Canvas "Model Development Guidelines" PDF (Model Pipeline assignment).
Course reference: [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps) labs and Reading Materials (Sessions 8, 10, 12).

Status as of 2026-11-12. ✅ done · 🟡 partly done · ⬜ not started. Nothing has been pushed to Git or GCP yet.

## 1. Requirements mapped to our work

| # | PDF requirement | How the course teaches it | SkinSort status |
|---|---|---|---|
| 1 | Load versioned data from the data pipeline | DVC on GCS (`dvc pull`), DVC_Labs/Lab_1 | ✅ `src/training/dataset.py` reads the DVC-tracked `Data-Pipeline/data/processed` output; every MLflow run is tagged with the DVC hashes from `Data-Pipeline/dvc.lock` (`data_dataset_md5`, `data_images_md5`, `data_matches_lock`). |
| 2 | Train several architectures, code that selects the best | best run via `mlflow.search_runs(order_by=['metrics.auc DESC'])` (MLflow Lab2); baseline first (Session 10) | ✅ 3 backbones on an L4 GPU (`train.py`); `select_model.py` ranks them with `search_runs(order_by=...)` and the phone candidates the same way, against 3 baselines. |
| 3 | Validation on a hold-out set with task metrics | hold-out split + metrics JSON (Github_Labs/Lab2 `evaluate_model.py`) | ✅ `evaluate.py` / `fit_final.py`: calibration + tau on validation only, test scored once; metrics.json, confusion matrix, calibration plot. |
| 4 | Bias detection by slicing (Fairlearn / TFMA) | Slides only: slicing, TFMA, Fairness Indicators (Session 12) | ✅ `bias.py`: Fairlearn `MetricFrame`, urgent recall (TPR) gap per slice, 95% intervals. |
| 5 | Bias report + mitigation, documented | Slides: thresholds, re-sampling | ✅ Reports + charts; per-group threshold mitigation on both the image model and the selected phone model (sex and skin-tone gaps closed, acral not), with trade-offs (§3 B). Re-weighting of the fusion head (CPU) not run — documented as next step. |
| 6 | Final model selection after bias checking | — | ✅ `select_model.py`: validation filters → `search_runs` ranking → bias consideration (§3 G). |
| 7 | Push model to GCP Artifact Registry / Model Registry | `register_model` + `transition_model_version_stage("Production")` (MLflow Lab2); versioned model + version file in GCS, image to Artifact Registry (Github_Labs/Lab4, `ci_cd_pipeline.yml`) | 🟡 Local MLflow registry: `skinsort-triage` v2 Production, v1 `previous` (gate + rollback). Code for GCS push (`registry push`, version.txt) and Artifact Registry push (CI) is written and tested with fakes; **not run** — waiting for the team's OK. |
| 8 | Hyperparameter tuning, document search space | Hyperopt TPE, nested MLflow runs (MLflow Lab2) | ✅ `tune.py`; re-run on the final backbone (`outputs/tuning_b3/`), tuned head tied out-of-fold → params.yaml settings kept (report §6). |
| 9 | Tracking + comparison plots and why the final model won | MLflow; Session 8 "compare experiments, consider latency, cost, fairness" | ✅ `outputs/selection/backbone_comparison.png` (AUC + size/CPU speed), `phone_comparison.png`, `model_selection.md`. |
| 10 | Sensitivity analysis: SHAP / LIME + hyperparameters | Slides (Session 12) | ✅ SHAP + hyperparameter sensitivity; plus robustness and threshold analysis (`analysis.py`, Session 12). |
| 11 | CI/CD: train, validate, bias, registry, notify, rollback | root `ci_cd_pipeline.yml`, Github_Labs Lab2/Lab4 | 🟡 `.github/workflows/model-pipeline-ci.yml` written (tests on every push; full pipeline on main); not run on GitHub yet (not pushed; needs secrets). Tests pass inside the Docker image (clean Linux). |
| 12 | Model development code in Docker | Multi-stage Dockerfile + compose (Docker_Labs/Lab2) | ✅ `docker/Dockerfile` (trainer → serving), `docker/docker-compose.yml`; both images built locally, serving container answers `/triage` with the same scores as the evaluation. |

Not covered by the course materials (we choose the tool ourselves): Fairlearn code, SHAP/LIME code, Slack/email from GitHub Actions, automated rollback.

## 2. Early results (2026-11-04 to 11-11)

Main dataset (dermoscopy), val split, best epoch per backbone — L4 GPU run on 2026-11-04:

| Backbone | Urgent AUC | Urgent recall | Macro F1 |
|---|---|---|---|
| efficientnet_b3 | 0.932 | 0.812 | 0.776 |
| resnet50 | 0.929 | 0.895 | 0.754 |
| convnext_tiny | 0.877 | 0.851 | 0.626 |

Phone photos (PAD-UFES-20), `pad_val` (345 photos, 31 urgent):

| Model | Urgent AUC |
|---|---|
| efficientnet_b3, image only (frozen) | 0.656 |
| Fusion on frozen efficientnet_b3, 10 epochs | 0.715 |
| Fusion on frozen efficientnet_b3, best of 50 epochs | 0.778 |
| efficientnet_b3 fine-tuned on `pad_train` (lr 3e-5, batch 8, best epoch 4), image only | 0.788 |
| **Fusion on the fine-tuned efficientnet_b3, best of 50 epochs (epoch 48)** | **0.808** |

Threshold check for 0.85 urgent recall on `pad_val` (27 of 31 urgent caught):

| Model | tau | Photos flagged urgent | Precision |
|---|---|---|---|
| Fusion on frozen efficientnet_b3 | 0.132 | 57% | 0.14 |
| Fusion on fine-tuned efficientnet_b3 | 0.108 | 41% | 0.19 |

Fine-tuning on phone photos cut the flagged share from 57% to 41%. Still high for a triage queue; with only 31 urgent cases the recall at that tau could plausibly be 0.74–0.97 on new photos.

At this point the test splits had not been scored yet (they were scored from 2026-11-11 on; see the disclosure in the report, section 4).

## 3. Plan

Ordered so each step feeds the next. Tools follow the course where it has a lab; otherwise the PDF's named tool.

**A. Finish the models**
1. ✅ Phone-photo fine-tune and fusion retrain done (`outputs/models/efficientnet_b3_pad/`, `outputs/models/fusion/best.pt`; frozen-backbone fusion kept as `fusion/best_frozen_b3.pt`).
2. ✅ `src/training/evaluate.py` written (calibration = temperature + per-class offset fitted on validation, tau picked on validation, test scored with both; writes `outputs/evaluation/<name>/` metrics.json, predictions.csv, confusion_matrix.png, calibration.png; logs to MLflow; `--fail-on-target` for CI). First results:

   | Model → test set | Urgent recall (target) | Precision | Flagged | AUC | Calibration error (≤0.05) |
   |---|---|---|---|---|---|
   | efficientnet_b3 → `test` (2,868, 630 urgent) | 0.873 (0.90) ❌ | 0.597 | 32% | 0.934 | 0.031 ✅ |
   | fusion on fine-tuned b3 → `pad_test` (339, 35 urgent) | 0.743 (0.85) ❌ | 0.144 | 53% | 0.669 | not meaningful — see below |

   - Image model: tau reached 0.901 recall on `val` but 0.873 on `test` — a small val→test gap. Options: pick tau with a margin on val, or improve the model; do not tune on test.
   - **Fine-tuned model defect found:** for ~1–2% of phone photos (6/345 in `pad_val`, 19/1601 in `pad_train`) the batch-8 fine-tuned efficientnet_b3 outputs exploding image features (up to ~15,000 vs a normal max of ~2), on CPU and MPS alike. The photos themselves look normal. The original GPU model has the same weakness on fewer photos (1/345). These photos get extreme, confidently wrong predictions, so no calibration can fit the fusion model (the best temperature runs to the top of the search grid). Fix (done, next bullet): re-run the fine-tune with BatchNorm layers frozen, retrain fusion, evaluate once more.
   - **Re-run with BatchNorm frozen (`--freeze-bn`), 2026-11-11:** exploding features in `pad_val` went from 6/345 to 0/345 (pad_train 19 → 9), and calibration now fits (temperature 1.26). The fine-tune itself was unstable: training loss jumped from 0.42 to 207 at epoch 4, so early stopping kept epoch 2 (image-only pad_val AUC 0.743, lower than the 0.788 of the unfrozen run). Fusion on it: pad_val AUC 0.805.

     | Fusion (frozen-BN fine-tune) → `pad_test` (339, 35 urgent) | Value | Target |
     |---|---|---|
     | Urgent recall | 0.829 (29/35) | 0.85 ❌ |
     | Calibration error | 0.046 | ≤ 0.05 ✅ |
     | Precision / flagged | 0.161 / 53% | |
     | AUC | 0.717 (pad_val 0.805) | |

     One more urgent case caught would pass the target, and with 35 urgent cases the result is noisy. The remaining weakness: a few photos still drive the original model's activations very high (max 475 in pad_train), which likely caused the loss spike. Next fixes to try: gradient clipping during fine-tuning, a larger batch on the GPU VM when an L4 is available, and normalising image features before the fusion head.
   - The test splits were scored three times while fixing evaluate.py bugs (calibration that diverged). Tau and calibration were always fitted on validation only; nothing was tuned on test results.

**B. Bias (PDF §2.4–2.5, §6)**
3. ✅ `src/training/bias.py` (2026-11-11). Reads `outputs/evaluation/<name>/predictions.csv` (evaluate.py now also saves the val rows), slices with Fairlearn `MetricFrame`, writes `bias_report.{json,md}`, one bar chart per slice and logs run `bias-<name>` to MLflow. `--fail-on-gap` gives exit code 1 for CI. Result is PASS, FAIL or INCONCLUSIVE (no slice had enough urgent cases).

   Image model (efficientnet_b3, tau 0.173 from val) — **FAIL**:

   | Split / slice | Urgent recall gap | Worst group (recall, urgent n) | Best group |
   |---|---|---|---|
   | test / sex | 0.015 ✅ | female 0.860 (172) | male 0.875 |
   | test / age_group | 0.202 ❌ | 0-30: 0.714 (21) · 30-45: 0.750 (60) | 75+ 0.916 |
   | test / body_site | 0.259 ❌ | acral 0.667 (21) | upper_extremity 0.926 |
   | test / source | 0.127 ❌ | ham10000 0.776 (152) | isic 0.904 |
   | fairness_test / skin_tone | 0.174 ❌ | III-IV 0.493 (203) | V-VI 0.667 (66) |

   - `test` has almost no Fitzpatrick labels (13 rows), so skin tone can only be checked on `fairness_test` (fitzpatrick17k clinical photos). There the model is out of domain (trained on dermoscopy): overall urgent recall ~0.55 and precision ~0.1–0.2 for every skin tone. The worst group is medium, not dark skin, so the main problem is the photo type, not skin tone alone.
   - Young patients and acral sites have few urgent cases (21 each; 95% interval for 0.714 is roughly 0.50–0.86), but the young-patient gap shows on val too (0-30: 0.69 on val).
   - Phone-photo fusion model (`fusion_pad`): **INCONCLUSIVE** — no group in `pad_test` has 20 urgent cases, so no gap can be measured.

4. 🟡 Mitigation: per-group thresholds (`--mitigate`), fitted on val rows only, never above the global tau (so no group loses recall):

   | Slice | Gap before → after | Flagged before → after | Result |
   |---|---|---|---|
   | source | 0.127 → 0.002 | 32.1% → 36.6% | ✅ fixed |
   | age_group | 0.202 → 0.209 | 32.1% → 33.5% | ❌ 0-30 has only 16 urgent val cases, keeps global tau |
   | body_site | 0.259 → 0.259 | 32.1% → 33.9% | ❌ acral has 4 urgent val cases |
   | skin_tone (fairness, fit on half the patients, reported on the other half) | 0.102 → 0.165 | 47.6% → 83.7% | ❌ worse; thresholds cannot fix out-of-domain photos |

   - Extra experiment: one threshold for all patients under 45 (49 urgent val cases, tau 0.088) lifts 0-30 recall to 0.905 and overall recall to 0.883 (flagged 34.5%), but 30-45 stays at 0.783, so age still fails (gap 0.133).
   - Next options: re-weight young / acral / ham10000 cases in training (`--class-weights`-style sample weights); add clinical photos (fitzpatrick17k train part) or pad photos to training for the skin-tone gap; collect more val cases for small groups.
5. ✅ Final model selection: done in `select_model.py` (§3 G). No candidate passes the bias check, so the best eligible one (fusion_oof) is selected with a bias warning; mitigation for it is below.

**C. Tuning and sensitivity (PDF §3, §5)** — done 2026-11-11
6. ✅ `src/training/tune.py`: Hyperopt TPE (course MLflow Lab2 pattern: parent run `tune-fusion`, one nested run per trial) on the fusion head over the frozen-BN `efficientnet_b3_pad` features (cached in `outputs/features/`). Each trial = 5-fold cross-validation on pad_train, grouped by patient (963 patients), so pad_val and pad_test are not used in the search. Search space: lr 1e-5–1e-3 (log), weight_decay 1e-5–1e-2 (log), dropout 0–0.5, urgent class weight 2–6, hidden_dim {32, 64, 128}.

   | | CV urgent AUC (pad_train) | pad_val AUC | pad_test recall @ tau | pad_test AUC | Calibration error |
   |---|---|---|---|---|---|
   | params.yaml settings (lr 1e-4, dropout 0.3, hidden 64, urgent weight 3) | 0.857 ± 0.018 | 0.805 | **0.829** | 0.717 | 0.046 ✅ |
   | Best trial 17 (lr 4.3e-4, wd 4.4e-3, dropout 0.16, urgent weight 5.3, hidden 128) | 0.867 ± 0.020 | 0.876 | 0.543 ❌ | 0.730 | 0.053 ❌ |

   - The tuned head ranks a bit better (CV +0.010, within one std; pad_test AUC +0.013), but its threshold does not transfer: the median P(urgent) of urgent photos is 0.21 on pad_val but 0.09 on pad_test, so tau picked on pad_val catches only 19 of 35 urgent photos on pad_test. With the higher learning rate and the best epoch picked on pad_val's 31 urgent cases, the head fits pad_val too closely.
   - **Decision: keep the params.yaml fusion head (`outputs/models/fusion/best.pt`) as the phone model.** Tuned model kept at `outputs/models/fusion_tuned/` for the report. Note: pad_test has now been used to compare these two candidates; report that honestly.
   - Fix to try later: pick tau from out-of-fold CV predictions on pad_train + pad_val (209 urgent cases instead of 31), and pick the final epoch count from CV (`cv_best_epoch_median`) instead of pad_val.
7. ✅ Sensitivity.
   - Hyperparameters (`outputs/tuning/sensitivity.png`, `search_progress.png`, Spearman with CV AUC): **learning rate matters most (+0.82)** — every trial with lr < 5e-5 scored ≤ 0.85. dropout (−0.25), hidden_dim (+0.21), weight_decay (+0.19) and urgent weight (+0.14) matter little.
   - SHAP (GradientExplainer on P(urgent), 200 pad_train background, all 345 pad_val photos; `outputs/explain/<model>/`). Kept model (`fusion`):

     | Input | Mean \|SHAP\| | Direction |
     |---|---|---|
     | Image (all features summed) | 0.130 | — |
     | q_changed (+ missing flag) | 0.013 (+0.014) | "yes" raises P(urgent) most (+0.057) |
     | q_grew (+ missing flag) | 0.013 (+0.012) | yes raises |
     | q_bleed | 0.011 | yes **lowers** P(urgent) — worth a clinical check (likely bleeding is common in the BCC cases labelled priority in PAD-UFES-20) |
     | q_elevation | 0.010 | yes raises |
     | q_itch, q_hurt | < 0.005 | little effect |

     The image drives most of the score; "has it changed" and "has it grown" are the most useful answers, matching the ABCDE "evolving" rule. Missing-answer flags for itch/bleed/elevation are never on in pad_val, so they get no SHAP value.

**C2. Threshold that transfers (`src/training/fit_final.py`), 2026-11-11**

Why: tau picked on pad_val (31 urgent photos) is noisy, and the tuned head fitted pad_val. A second problem showed up: `efficientnet_b3_pad` was fine-tuned on pad_train, so its features are easier on pad_train (image-only urgent AUC 0.839) than on new photos (0.743 on pad_val). That makes any score on pad_train too optimistic, and it also inflated the tuning CV scores (~0.86).

How: the original GPU `efficientnet_b3` (trained only on dermoscopy `train`) gives honest features for all phone photos. pad_train + pad_val are pooled (1,946 photos, 209 urgent) and split into 5 folds grouped by patient. Epoch count = median best epoch across folds (44). Each fold gets a head trained for exactly 44 epochs on the other folds; all pooled photos are scored this way (out-of-fold). Calibration and tau are fitted on those 1,946 out-of-fold scores, then a final head is trained on all pooled photos. Temperature, offsets and tau are saved inside the checkpoint (`outputs/models/fusion_oof/best.pt`). `fit_final.py` refuses a backbone whose `trained_on` overlaps the pooled splits; `train.py` now records `trained_on` in new checkpoints.

Settings chosen without test data: OOF urgent AUC was 0.774 for params.yaml settings vs 0.772 for the tuned settings, so params.yaml.

| Phone model | Threshold from | Honest estimate (OOF, 209 urgent) | pad_test recall | pad_test AUC | Calibration error |
|---|---|---|---|---|---|
| `fusion` (b3_pad backbone) | pad_val, 31 urgent | n/a (backbone saw pad_train) | 0.829 (29/35) | 0.717 | 0.046 |
| `fusion_tuned` | pad_val, 31 urgent | n/a | 0.543 (19/35) | 0.730 | 0.053 |
| **`fusion_oof`** (original b3) | out-of-fold, 209 urgent | recall 0.852, AUC 0.774, flagged 51% | 0.743 (26/35) | 0.707 | **0.037 ✅** |

- **pad_test is not like the other splits:** 34 of its 35 urgent photos are SCC and only 1 is melanoma; pad_train/pad_val are ~25% melanoma. SCC is the hardest case (it looks like BCC / actinic keratosis), so every model scores AUC ~0.71–0.73 on pad_test vs ~0.77–0.81 elsewhere. fusion_oof catches 25/34 SCC on pad_test (95% interval 0.57–0.85) vs 0.85 on SCC out-of-fold — the gap is within chance for 35 cases. The data pipeline should re-split the phone photos stratified by diagnosis.
- The three pad_test results (29, 19, 26 of 35) are within each other's noise. Picking the model with the best pad_test recall would be tuning on test. Recommendation: use `fusion_oof`, whose threshold comes from 209 urgent cases with no leakage and whose calibration passes; report its pad_test miss honestly.

Bias on fusion_oof (`bias.py --include-val` on the out-of-fold rows, which have enough urgent cases per group): sex gap 0.108 (female 0.794 vs male 0.902) ❌, body_site 0.130 (acral 0.750) ❌, skin_tone 0.052 (I-II 0.841 vs III-IV 0.893) ❌ just over, age 0.043 ✅. Dark skin has only 9 photos (2 urgent).

**Mitigation on the selected phone model (fusion_oof), 2026-11-12** — `bias.py --name fusion_oof --include-val --mitigate sex body_site skin_tone`, output `outputs/bias/fusion_oof/mitigation.md`.

How: per-group thresholds (PDF §6.3 "adjusting decision thresholds for different groups"), never above the global tau 0.070. Fitted with two-fold cross-fitting by patient on the 1,946 out-of-fold photos: thresholds fitted on one half of the patients are applied to the other half and vice versa, so no photo is judged by a threshold fitted on it. A group needs ≥ 20 urgent photos in the fitting half to get its own threshold.

| Slice | Gap before → after | Group change | Flagged (all photos) | Result |
|---|---|---|---|---|
| sex | 0.108 → **0.046** | female tau 0.054: recall 0.794 → 0.856; female photos flagged 62% → 73% | 51.2% → 54.7% | ✅ PASS |
| skin_tone | 0.052 → **0.045** | light skin tau ~0.064: recall 0.841 → 0.848 | 51.2% → 52.0% | ✅ PASS |
| body_site | 0.130 → 0.130 | acral has 24 urgent photos, ~12 per half: too few to fit its own threshold | 51.2% → 52.7% | ❌ unchanged |

Trade-offs:
- Closing the sex gap costs about 3.5 points more flagged photos overall (11 points more among women): more doctor review time, no missed urgent cases added for anyone (thresholds only go down).
- A per-group threshold needs the group at prediction time. The intake today sends photo + 6 answers only. Sex and body site can be asked at intake; skin tone cannot be asked reliably. So the sex threshold is deployable after an intake change; the skin-tone one is not. The registered model (v2) still uses the single global tau; the group thresholds are evaluated offline and documented.
- Acral gap needs more acral urgent cases (data collection), not a threshold.
- On pad_test (report only; 35 urgent): female recall 0.810 → 0.952 with the val-fitted threshold; male stays 0.643 (14 urgent photos — too few to read much into).
- Re-weighting the fusion head (CPU, ~1 hour) is the next option if the acral gap must be addressed in the model; not run.

**Dataset artifact found (PAD-UFES-20):** photos without patient demographics (794 photos, sex "unknown") contain **no** urgent cases — they are mostly non-biopsied actinic keratoses. And photos with the "changed"/"grew" answer missing are 2.5× more likely urgent (21% vs 8.5%). The model uses the missing flags (SHAP: `q_changed_missing` 0.028, about as much as `q_bleed`), so part of its answer signal is how the dataset was collected, not what patients say. To try: drop the missing flags (or fill missing as "no") and compare out-of-fold AUC.

SHAP for fusion_oof (pad_val rows, which this head trained on): image 0.103; q_changed 0.035 (yes → +0.15 on P(urgent)); q_grew 0.035; q_bleed 0.029 (yes lowers); missing flags 0.028 / 0.024.

**D. Registry (PDF §2.6, §8.7)** — 2026-11-11
8. ✅ `src/training/registry.py` + `src/serving/triage_model.py`.
   - The registered model is an MLflow pyfunc (models-from-code file, so nothing is pickled) that does the whole triage step: photo path + 6 answers in → calibrated P(routine/priority/urgent), urgent flag (P(urgent) ≥ tau) and triage class out. It bundles the image backbone, the fusion head (with temperature, offsets, tau) and params.yaml. Its output matches `outputs/evaluation/fusion_oof/predictions.csv` exactly.
   - Gate (`params.yaml` `registry:`): `calibration_error` must pass; `urgent_recall` and `bias` are recorded as warnings on the version; test urgent AUC may not drop more than 0.01 below the current Production version. Promotion moves the old Production version to alias `previous`; `rollback` swaps them back. The course's `transition_model_version_stage("Production")` is also called (deprecated in MLflow 3, kept for the course pattern).
   - Result: **`skinsort-triage` v1 = fusion_oof, promoted to Production** (no earlier version). Version tags: `recall_target_met=false`, `calibration_ok=true`, `bias_result=FAIL`, warnings "urgent_recall not met; bias not met".
9. 🟡 `registry.py push`: copies the Production version to `gs://skinsort-data-asdr/models/skinsort-triage/v<N>/` plus `model_info.json` (version, run id, tags, git commit, time) and updates `latest.json`; never overwrites an existing version (course Github_Labs/Lab4 pattern, `google-cloud-storage`). Dry run: 32 files, 43.9 MB. Not uploaded yet. The Artifact Registry image is built in step E (Dockerfile).

**G. Model selection after bias checking (`src/training/select_model.py`), 2026-11-12**

Course pattern: every candidate is an MLflow run, ranked with `mlflow.search_runs(order_by=[... DESC])`. Baselines first (Session 10), deployment cost considered (Session 10 p.21, Session 8 p.19). Test metrics are logged for the report only.

- Stage 1, backbones (dermoscopy val):

  | Backbone | Val urgent AUC | Params | Checkpoint | CPU ms/photo |
  |---|---|---|---|---|
  | **efficientnet_b3** ✅ | 0.932 | 10.7M | 43 MB | ~200 |
  | resnet50 | 0.929 | 23.5M | 94 MB | ~35–42 |
  | convnext_tiny | 0.877 | 27.8M | 111 MB | ~370 |

  efficientnet_b3 wins on AUC (by 0.003 — effectively a tie with resnet50) and is half the size, but **resnet50 is ~5× faster on CPU** (depthwise convolutions are slow on CPU). For a CPU server resnet50 is a fair alternative; we keep the AUC rule.
- Stage 2, phone candidates on the pad_val photos. Filters: validation calibration error ≤ 0.05; tau fitted on ≥ 100 urgent photos; servable. Bias: a bias-passing candidate within 0.02 AUC of the best is preferred.

  | Candidate | Val AUC | Tau from | Bias | Result |
  |---|---|---|---|---|
  | fusion_pad_tuned | 0.869 | 31 urgent | inconclusive | filtered (tau support) |
  | **fusion_oof** | 0.813 | 209 urgent | FAIL | **selected**, with bias warning |
  | fusion_pad | 0.806 | 31 urgent | inconclusive | filtered (tau support) |
  | baseline_answers_lr | 0.781 | 209 | FAIL | baseline |
  | baseline_image_only | 0.663 | 209 | FAIL | baseline |
  | baseline_prevalence | 0.500 | 209 | PASS (flags everyone) | baseline |

  The image adds only ~0.03 AUC over the answers alone, and the image alone (no phone fine-tuning) is weak (0.663).

**H. Extra analysis (`src/training/analysis.py`, `outputs/analysis/`), 2026-11-12**

- **Missing-answer artifact, measured** (out-of-fold, no test data): zeroing the 6 missing flags drops fusion AUC 0.774 → 0.710 and answers-only 0.750 → 0.655. On photos with **all answers filled in** (what real use looks like) fusion_oof has out-of-fold recall **0.788** at tau (not 0.852), flagging 43%; it flags 88% of photos with a missing answer. The honest deployed estimate is therefore recall ≈ 0.79. Fix to try: train without the missing flags and collect complete answers.
- **Robustness on pad_test** (photos degraded before scoring; fixed tau): clean AUC 0.707 / recall 0.743; blur 0.643 / 0.686; dark 0.716 / 0.771; bright 0.693 / 0.800; JPEG q15 0.657 / 0.886 but flags 74%; 4× lower resolution 0.738 / 0.800. Blur hurts ranking most; heavy compression makes the model flag almost everything. A photo-quality check in the intake app would help.
- **Threshold trade-off** (`thresholds.png`): out-of-fold, recall 0.85 needs ~51% of photos flagged; recall 0.75 needs ~40%.

**E. Automation (PDF §7, §8.1)** — written 2026-11-12, not pushed
10. ✅ `docker/Dockerfile` (base → `trainer` runs `src.training.pipeline`; `serving` = FastAPI `src/serving/app.py` with the exported Production model) and `docker/docker-compose.yml` (trainer, then serving after it completes, like Docker_Labs/Lab2). Built locally; `/health` and `/triage` checked.
11. 🟡 `.github/workflows/model-pipeline-ci.yml`: tests on every push/PR; on main: GCP auth → `dvc pull` → backbone from GCS → `src.training.pipeline --fail-if-rejected` → `registry compare-deployed` (worse than the deployed model = nothing pushed) → bias alert email → GCS push → Artifact Registry image → email (`src/notify.py`, smtplib like the Airflow lab). Needs secrets `GCP_SA_KEY`, `GCP_PROJECT_ID`, `SMTP_USER`, `SMTP_PASSWORD`, `NOTIFY_TO`, an Artifact Registry repo `skinsort` in us-central1, and the backbone uploaded once to `gs://skinsort-data-asdr/models/backbones/efficientnet_b3/best.pt`.

**F. Wrap-up**
12. ✅ README model pipeline section; versions pinned in `requirements-model.txt` (`mlflow==3.18.0`, Python 3.11). ⬜ Commit, push, PR, GCS / Artifact Registry upload — waiting for the team's OK.

**I. Audit fixes, 2026-11-12 (after an independent review against the assignment)**
- Serving now applies the data pipeline's preprocessing to raw uploads (`src/data/preprocess.py`; before, it only resized, unlike every training image).
- Registry gate on validation / out-of-fold metrics only (test results are warnings); GCS has its own version numbering (`version.txt`) and `rollback-gcs`.
- Reweighing (Kamiran & Calders, same formula as the data pipeline) → `fusion_oof_rw`: acral and skin-tone gaps closed, sex gap 0.108 → 0.069 (0.039 with a per-sex threshold). Selection rule refined to "fewest failing bias slices within 0.02 AUC" (disclosed in the report); `fusion_oof_rw` selected, registry v3 from a clean 6-step pipeline run.
- lr one-at-a-time sweep; data-version tags on all pipeline runs; key charts copied to `docs/figures/`; CI: skip without secrets, lint, `mlruns/` artifact, dvc.lock trigger, selection-driven gate; compose uses `serving_base`; README commands fixed.
