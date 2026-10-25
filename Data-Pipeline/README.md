# SkinSort — Data Pipeline

Airflow pipeline that downloads, validates, preprocesses and versions the four
SkinSort datasets (ISIC Archive, HAM10000, PAD-UFES-20, Fitzpatrick17k).

> **Status:** acquisition, preprocessing, orchestration, logging/alerts, DVC
> versioning and tests are in place. Schema & statistics (Great Expectations),
> anomaly alerts and bias slicing are the next stages and will join the same DAG.

## Pipeline (current DAG)

```
ingest_isic ─────────────┐
ingest_ham10000 ─────────┤                ┌─ version_raw_data
ingest_pad_ufes_20 ──────┼─ ingest_report ┤
ingest_fitzpatrick17k ───┘                └─ harmonize ─ image_stats ─ quality_filter
                                              ─ dedupe ─ transform ─ split ─ version_processed_data
```

| Task | What it does | Output |
|---|---|---|
| `ingest_isic` | Pages the ISIC API v2 per diagnosis group (caps in `params.yaml`), downloads images in parallel | `data/raw/isic/` |
| `ingest_ham10000` | Harvard Dataverse API, verifies MD5 of each zip, extracts images | `data/raw/ham10000/` |
| `ingest_pad_ufes_20` | Mendeley API, verifies SHA-256 of each zip, extracts images | `data/raw/pad_ufes_20/` |
| `ingest_fitzpatrick17k` | GitHub CSV, then each image from its atlas URL; verifies each image against the MD5 in the CSV; skips links that were 404 last run | `data/raw/fitzpatrick17k/` |
| `ingest_report` | Counts per dataset | `logs/ingest_report.json` |
| `version_raw_data` | `dvc add data/raw` + `dvc push` | `data/raw.dvc` |
| `harmonize` | One common schema for all datasets; diagnosis → canonical label → urgency tier | `data/interim/harmonized.csv` |
| `image_stats` | Opens every image once: size, brightness, contrast, sharpness, perceptual hash (process pool) | `data/interim/image_stats.csv` |
| `quality_filter` | Drops unreadable, too small, too dark/bright, flat or blurry photos | `quality_passed.csv`, `logs/quality_report.json` |
| `dedupe` | Removes ISIC↔HAM copies and cross-pool near-copies; merges near-copies into one split group | `deduped.csv`, `duplicates_dropped.csv` |
| `transform` | Centre-crop + resize, Shades-of-Gray colour constancy; encodes age/sex/site/answers with missing flags | `data/processed/images/`, `features.csv` |
| `split` | 70/15/15 by patient/lesion group, stratified by urgency; PAD has its own split; Fitzpatrick17k = fairness test only; fails on any leakage | `data/processed/dataset.csv`, `logs/split_report.json` |
| `version_processed_data` | `dvc commit` + `dvc push` of all stage outputs | `dvc.lock` |

Downloads run in parallel. Every task retries twice with exponential back-off
and sends an alert when it finally fails. The same steps are declared in
`dvc.yaml`, so `dvc repro` rebuilds the processed data without Airflow.

## Preprocessing decisions

* **Labels.** `configs/harmonization.yaml` maps each dataset's labels to one
  canonical diagnosis; `configs/urgency_mapping.yaml` maps diagnoses to
  routine / priority / urgent (draft, to be checked against a clinical source).
  The step fails if any diagnosis has no tier, so nothing is silently unlabelled.
* **HAM10000 `akiec`** mixes actinic keratosis with Bowen's disease (SCC in situ)
  and cannot be separated; it is mapped to *priority* and flagged for review.
* **Missing patient answers.** Only PAD-UFES-20 has answers (itch, grew, hurt,
  changed, bleed, elevation). Everyone else gets `q_*_missing = 1` instead of a
  made-up value. Duration and "other similar spots" (from the scoping doc's four
  questions) exist in no public dataset.
* **Leakage.** Splits are by patient, else lesion, else image. Near-duplicate
  photos (pHash ≤ 4 bits apart) are merged into one group; copies across the
  dermoscopic / PAD / Fitzpatrick pools are removed. `split` re-checks that no
  group or hash appears in two splits and fails otherwise.
* **Augmentation** (blur, crop, brightness to mimic phone photos) happens at
  training time, not here, so each epoch sees new variants.

## Pipeline flow optimisation (Airflow Gantt)

**Run 1 — downloads only, 48 min 50 s** ([full screenshot](docs/gantt_run1_before.png))

![Gantt chart, run 1](docs/gantt_run1_before.png)

`ingest_fitzpatrick17k` took **41 min 35 s** while the other three downloads
finished in 3–5 min, and `ingest_report` / `version_raw_data` had to wait for it.
Its log showed why: ~12,600 of its 16,536 image links (all on dermaamin.com)
return HTTP 404, and each dead link takes ~8 s to answer.

**Fixes**

| Bottleneck | Fix | Effect |
|---|---|---|
| Dead Fitzpatrick17k links re-tried every run | Links that returned 404 are remembered in `failures.csv` and skipped (`skip_known_404`); timeouts and 5xx are still retried | 41.6 min → **0.2 min** |
| Every download re-fetched | Per-dataset `manifest.csv` with checksums; files already present are skipped | ISIC 2.9 → 0.4 min, HAM 3.1 → 0.1, PAD 4.7 → 0.1 |
| Four independent downloads | Run as parallel Airflow tasks; images within a dataset use a 16-thread pool | wall time = slowest download, not the sum |
| `version_raw_data` blocking preprocessing | Runs in parallel with `harmonize` instead of before it | ~1.7 min off the critical path |
| Full-resolution decode of 4,900 px ISIC JPEGs | JPEG draft-mode decode at reduced scale in `image_stats` / `transform`; both use a process pool over all cores | ~1.6× faster per image in a local benchmark |

**Run 2 — downloads + full preprocessing, 21 min 56 s** ([full screenshot](docs/gantt_run2_after.png))

![Gantt chart, run 2](docs/gantt_run2_after.png)

Run 2 does far more work (7 extra preprocessing tasks) in under half the time.
The remaining long bars are `image_stats` (8.5 min) and `transform` (9.6 min):
CPU-bound image work on the VM's 4 cores. Next options: a larger VM for
preprocessing runs, or merging both passes so each image is decoded once.

## Folder structure

```
configs/                             # (repo root) harmonization + urgency mapping
Data-Pipeline/
├── dags/skinsort_data_pipeline.py   # Airflow DAG
├── scripts/
│   ├── common/                      # config, logging, HTTP, files, alerts
│   ├── ingest/                      # one module per dataset
│   ├── preprocess/                  # harmonize, image_stats, quality, dedupe, transform, split
│   └── versioning.py                # DVC add / commit / push
├── tests/                           # pytest unit tests (network fully faked)
├── data/                            # raw/, interim/, processed/ (DVC-tracked, not in Git)
├── logs/                            # pipeline.log, *_report.json, Airflow logs
├── params.yaml                      # every tunable value
├── dvc.yaml                         # same steps as the DAG, for `dvc repro`
├── Dockerfile, docker-compose.yaml  # Airflow 3.3.2 with LocalExecutor
└── requirements.txt, requirements-dev.txt
```

## Setup

### Option A — run on the team GCP VM (recommended)

The VM `skinsort-airflow` (us-east1-b) already has Docker and this repo.
It uses the `skinsort-pipeline` service account, so no key file is needed.

```bash
gcloud compute instances start skinsort-airflow --zone=us-east1-b
# open the Airflow UI through an SSH tunnel (port 8080 is not public)
gcloud compute ssh skinsort-airflow --zone=us-east1-b -- -L 8080:localhost:8080
# then browse to http://localhost:8080   (login: see Data-Pipeline/.env on the VM)

# when finished — the VM costs ~$0.13/hour while running
gcloud compute instances stop skinsort-airflow --zone=us-east1-b
```

### Option B — run locally with Docker

```bash
gcloud auth application-default login          # lets DVC/GCS use your account
cd Data-Pipeline
cp .env.example .env                           # set AIRFLOW_UID=$(id -u) on Linux
docker compose up -d --build
# http://localhost:8080
```

### Development environment (tests, lint, running one step)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r Data-Pipeline/requirements-dev.txt
cd Data-Pipeline
pytest                    # unit tests, no network needed
flake8 .                  # PEP 8
python -m scripts.ingest.isic --limit 5              # quick real download test
python -m scripts.ingest.fitzpatrick17k --limit 50
```

## Running the pipeline

In the Airflow UI, un-pause **skinsort_data_pipeline** and press *Trigger*, or:

```bash
docker compose exec airflow-scheduler airflow dags trigger skinsort_data_pipeline
```

Downloads are idempotent: files already recorded in a dataset's
`manifest.csv` are skipped, so a failed run can simply be re-triggered.

## Data versioning (DVC)

* Remote: `gs://skinsort-data-asdr/dvc-store` (default remote `gcs`, see `.dvc/config`).
* `version_raw_data` updates `Data-Pipeline/data/raw.dvc`. Commit that pointer file
  so the data version is recorded in Git history:
  `git add Data-Pipeline/data/raw.dvc && git commit -m "data: <what changed>"`.
* Teammates get exactly the same data with `dvc pull`.
* Going back to an older data version: `git checkout <commit> -- Data-Pipeline/data/raw.dvc && dvc checkout`.

## Logging and alerts

* Every step logs to the Airflow task log **and** `logs/pipeline.log` (rotating).
* `logs/ingest_report.json` summarises each run (images listed, downloaded, failed).
* Failed downloads are listed in `data/raw/<dataset>/failures.csv`.
* Alerts: set `SLACK_WEBHOOK_URL` and/or `ALERT_EMAIL_TO` + `SMTP_*` in `.env`.
  Without them, alerts are still written to the log.

## Datasets and known issues

| Dataset | Source | Notes |
|---|---|---|
| ISIC Archive | `api.isic-archive.com/api/v2` | Capped subset per diagnosis group; IDs frozen in `metadata.csv`. The API's reported file size is ~3 KB smaller than the served JPEG, so size is not checked. |
| HAM10000 | Harvard Dataverse `doi:10.7910/DVN/DBW86T` | `image_id` values are ISIC IDs, which makes ISIC/HAM de-duplication exact. |
| PAD-UFES-20 | Mendeley `zr7vgbcyr2` v1 | Only dataset with patient answers (itch, grew, hurt, changed, bleed). |
| Fitzpatrick17k | `github.com/mattgroh/fitzpatrick17k` | **Fairness evaluation only, never redistributed.** As of Oct 2026 all `dermaamin.com` links (~76% of rows) return 404; only `atlasdermatologico.com.br` images (~3.9k) download. `min_success_rate` is set to 0.2 accordingly. |

Licences: HAM10000 and parts of ISIC are CC BY-NC — SkinSort is an academic,
non-commercial project. Each ISIC image's licence is kept in `metadata.csv`.
