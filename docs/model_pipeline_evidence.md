# Model pipeline: evidence that it runs

What the CI/CD pipeline produced on GitHub and Google Cloud (assignment §2.6, §7, §8.7). The code and the
reasoning are in [model_pipeline_report.md](model_pipeline_report.md); charts in [figures/](figures/README.md).

## CD runs on `main` (`.github/workflows/model-pipeline-cd.yml`)

| Date (UTC) | Trigger | Commit | Run | Result |
|---|---|---|---|---|
| 2026-11-15 23:11 | CI passed after the merge of PR #4 | `1100fb6` | [43773416059](https://github.com/DHEEKSHIT11/skinSort/actions/runs/43773416059) | success: bias FAIL (sex, one threshold) → bias alert step ran; deployed GCS v1 + `skinsort-serving:v1` |
| 2026-11-16 00:06 | CI passed after the merge of PR #7 (bias fix) | `3991f1a` | [43778352428](https://github.com/DHEEKSHIT11/skinSort/actions/runs/43778352428) | success: bias PASS; deployed GCS v2 + `skinsort-serving:v2` |
| 2026-11-16 00:33 | manual (`workflow_dispatch`) | `3991f1a` | [43781045652](https://github.com/DHEEKSHIT11/skinSort/actions/runs/43781045652) | success: bias PASS; deployed GCS v3 + `skinsort-serving:v3`, `latest`; Slack success message delivered |

Every run executed the same steps, all green:

| Step | What it checks or does | Assignment |
|---|---|---|
| Authenticate with GCP (keyless, Workload Identity Federation) | no JSON key; only this repository may act as `skinsort-pipeline` | §7.1 |
| Pull versioned data (DVC) and the GPU-trained backbone | `dvc pull` at the version in `Data-Pipeline/dvc.lock`; backbone from `gs://skinsort-data-asdr/models/backbones/` | §2.1, §8.2 |
| Train, validate, bias check, select, register (gate) | `python -m src.training.pipeline --fail-if-rejected`: out-of-fold training, validation metrics, Fairlearn slicing, selection after bias, MLflow registry gate (AUC ≥ 0.75, calibration ≤ 0.05) | §2.2–2.5, §7.2, §8.3–8.6 |
| Gate against the deployed model | `registry compare-deployed`: not pushed if worse than `latest.json` in GCS (rollback = keep the deployed version) | §7.6 |
| Bias alert | Slack alert when the selected model fails a slice | §7.3 |
| Push the model to GCS (versioned) | `registry push` → `v<N>/`, `latest.json`, `version.txt` | §2.6, §7.4 |
| Push the model to Artifact Registry (versioned package) | generic repository `skinsort-models`, package `skinsort-triage`, version `v<N>` (added after run 3; v3 uploaded by hand) | §2.6, §7.4, §8.7 |
| Build and push the serving image to Artifact Registry | `us-central1-docker.pkg.dev/skinsort-512623/skinsort/skinsort-serving:v<N>` | §7.4, §8.1 |
| Keep reports | selection, bias, evaluation reports, `mlflow.db`, `mlruns/` as run artifacts | §4 |
| Notify | Slack `#pipeline-alerts` on success or failure (`src/notify.py`) | §7.5 |

Key log lines of the latest run:

```
OOF all:      urgent AUC 0.769, recall 0.852, flagged 52.1%
bias check with one global threshold: FAIL ['val/sex']
bias check PASS
skinsort-triage version 1: PROMOTED to production
new model vs deployed v1: OK to deploy
bias result: PASS
NOTIFY: SkinSort model pipeline: success
```

(`version 1` is the fresh MLflow registry on the GitHub runner; the deployed number is the GCS version below.)

## Google Cloud (project `skinsort-512623`)

Model versions in Cloud Storage (`gs://skinsort-data-asdr/models/skinsort-triage/`):

```
latest.json      -> gcs_version 3, git_commit 3991f1a, bias_result PASS, group_thresholds "sex, age_group"
version.txt      -> 3
v1/  v2/  v3/    (MLflow pyfunc: MLmodel, triage_model.py, fusion + image checkpoints, params.yaml,
                  deployed_thresholds.json from v2 on)
```

Serving image in Artifact Registry (`us-central1-docker.pkg.dev/skinsort-512623/skinsort`):

| Image | Tags | Created (UTC) |
|---|---|---|
| skinsort-serving | `v3`, `latest` | 2026-11-16 00:50 |
| skinsort-serving | `v2` | 2026-11-16 00:31 |
| skinsort-serving | `v1` | 2026-11-15 23:26 |

Model packages in Artifact Registry (`us-central1-generic.pkg.dev/skinsort-512623/skinsort-models`):

| Package | Version | Content |
|---|---|---|
| skinsort-triage | `v3` | the deployed MLflow pyfunc model (same files as GCS `v3/`), 41 MB |

From the next CD run on, every deployed version is uploaded here by the workflow.

Other artifacts:
- `gs://skinsort-data-asdr/models/backbones/{efficientnet_b3,resnet50,convnext_tiny}/best.pt`: the GPU-trained image
  networks (checksums match the training outputs).
- `gs://skinsort-data-asdr/mlflow/`: backup of the development `mlflow.db` and `mlruns/` (all training runs and
  registered versions).

Model files are not in git (`.gitignore`: `outputs/`, `*.pt`, `mlruns/`, `mlflow.db`, `models/triage_model/`).

## CI (`.github/workflows/model-pipeline-ci.yml`)

Lint (flake8) and 76 unit tests on every push and pull request that touches the model code; green on PR #4, PR #7
and `main`. CD starts only after CI passes on `main`.

## Rollback, tested live

`python -m src.training.registry rollback-gcs --image us-central1-docker.pkg.dev/skinsort-512623/skinsort/skinsort-serving`
moves both the deployed model (GCS `latest.json`) and the serving image's `latest` tag. Run on 2026-11-16 01:14 UTC:

```
before:                 GCS latest.json -> v3    image tags: latest,v3  v2  v1
rollback (one back):    GCS latest.json -> v2 (rolled_back_from 3)    image tags: v3  latest,v2  v1
restore (--to 3):       GCS latest.json -> v3 (rolled_back_from 2)    image tags: latest,v3  v2  v1
```

The automatic part runs in every CD run: `registry compare-deployed` refuses to push a model whose validation AUC is
more than 0.01 below the deployed one (unit-tested in `tests/test_registry.py`).

## Docker: the whole training process runs in the trainer image

```
docker build -f docker/Dockerfile --target trainer -t skinsort-trainer .
docker run --rm --shm-size=2g -v "$PWD:$PWD" -w "$PWD" skinsort-trainer src.training.train --smoke --no-pretrained --backbone resnet18
  -> device cpu, 200 train images, 200 val images
     [resnet18] epoch 1/1  loss 1.033  val urgent AUC 0.505 ...   (smoke check, 2026-11-16)
docker run ... skinsort-trainer                     # default: src.training.pipeline (fusion -> bias -> registry)
```

For the full backbone training the same image is built with CUDA PyTorch (`--build-arg TORCH_INDEX=...`) and run on
the GPU machine (`docker compose --profile gpu run backbones`).
