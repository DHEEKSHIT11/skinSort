"""Model registry: package, register, promote (with gates) and roll back.

Follows the course MLflow lab pattern (log model -> register_model -> move to
Production) with two additions the assignment asks for: a promotion gate (a new
version only becomes Production if it passes the hard checks and is not worse than the
current Production version on the same test set) and rollback.

The registered model is an MLflow pyfunc that serves the whole triage step:
photo path + 6 patient answers in, calibrated P(routine / priority / urgent) and the
urgent flag (P(urgent) >= tau) out. It bundles the image backbone checkpoint, the
fusion head checkpoint (with its calibration and tau) and params.yaml.

    # register fusion_oof and promote it if it passes the gate
    python -m src.training.registry register --checkpoint outputs/models/fusion_oof/best.pt \\
        --evaluation outputs/evaluation/fusion_oof --bias outputs/bias/fusion_oof

    python -m src.training.registry status
    python -m src.training.registry rollback         # Production -> the previous version
    python -m src.training.registry push --dry-run   # copy Production to GCS as v<N>/

The "production" alias is the source of truth (MLflow 3 recommends aliases); the
classic "Production" stage is set as well because the course labs use stages.
Gate settings live in params.yaml (registry section).
"""
import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.training.dataset import ROOT, load_params, data_version
from src.training.train import setup_mlflow

PRODUCTION, PREVIOUS = "production", "previous"
MODEL_CODE = ROOT / "src" / "serving" / "triage_model.py"


def gate(candidate: Dict[str, Any], production: Optional[Dict[str, Any]],
         cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Promotion decision, on VALIDATION metrics only (metrics.json "val": out-of-fold photos
    for fit_final models, the validation split for evaluate.py models).

    Required: validation urgent AUC >= min_val_urgent_auc, validation calibration error <=
    max_val_calibration_error, and no drop of more than max_regression in validation urgent AUC
    versus the current Production version. Test results and the bias check are recorded as
    warnings on the version; they never decide promotion."""
    val, test = candidate["val"], candidate.get("test", {})
    checks = {
        "val_urgent_auc": {"value": val["urgent_auc"], "passed": val["urgent_auc"] >= cfg["min_val_urgent_auc"]},
        "val_calibration_error": {"value": val["calibration_error"],
                                  "passed": val["calibration_error"] <= cfg["max_val_calibration_error"]},
    }
    for name, check in candidate.get("checks", {}).items():          # test-set checks: report only
        checks[f"test_{name}"] = check
    if "bias_passed" in candidate:
        checks["bias"] = {"passed": candidate["bias_passed"]}
    reasons = [f"required check {n} failed" for n in cfg["required_checks"] if not checks.get(n, {}).get("passed")]
    warnings_ = [f"{n} not met" for n in cfg["warn_checks"] if n in checks and not checks[n]["passed"]]
    metric = cfg["compare_metric"]
    new = val[metric]
    old = production["val"][metric] if production is not None and "val" in production else None
    if old is not None and new < old - cfg["max_regression"]:
        reasons.append(f"validation {metric} {new:.3f} is worse than Production {old:.3f}")
    return {"promote": not reasons, "reasons": reasons, "warnings": warnings_, "metric": f"val_{metric}",
            "candidate_value": new, "production_value": old, "test_urgent_auc": test.get("urgent_auc")}


def log_triage_model(checkpoint: Path, image_example: Optional[pd.DataFrame] = None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "tau" not in ckpt:
        raise ValueError(f"{checkpoint} has no tau inside; build it with src.training.fit_final")
    if image_example is not None:
        # answers are floats in the signature: NaN = not answered (ints would reject NaN)
        image_example = image_example.copy()
        for col in image_example.columns.drop("image_path"):
            image_example[col] = image_example[col].astype(float)
    artifacts = {"fusion_checkpoint": str(checkpoint), "image_checkpoint": ckpt["image_checkpoint"],
                 "params": str(ROOT / "params.yaml")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # models-from-code: MLflow runs this file to build the model instead of pickling it
        return mlflow.pyfunc.log_model(name="triage_model", python_model=str(MODEL_CODE),
                                       artifacts=artifacts, code_paths=[str(ROOT / "src")],
                                       input_example=image_example,
                                       pip_requirements=str(ROOT / "requirements-model.txt"))


def version_metrics(client: MlflowClient, version) -> Dict[str, Any]:
    path = client.download_artifacts(version.run_id, "evaluation/metrics.json")
    return json.loads(Path(path).read_text())


def register(params: Dict[str, Any], checkpoint: Path, evaluation_dir: Path,
             bias_dir: Optional[Path] = None, image_example: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    cfg = params["registry"]
    name = params["mlflow"]["registered_model_name"]
    client = MlflowClient()
    candidate = json.loads((evaluation_dir / "metrics.json").read_text())
    bias = None
    if bias_dir and (bias_dir / "bias_report.json").exists():
        bias = json.loads((bias_dir / "bias_report.json").read_text())
        candidate["bias_passed"] = bias["result"] != "FAIL"

    try:
        current = client.get_model_version_by_alias(name, PRODUCTION)
    except MlflowException:
        current = None
    production = version_metrics(client, current) if current else None
    decision = gate(candidate, production, cfg)

    with mlflow.start_run(run_name=f"register-{checkpoint.parent.name}") as run:
        mlflow.set_tags(data_version(params))
        info = log_triage_model(checkpoint, image_example)
        mlflow.log_artifacts(str(evaluation_dir), "evaluation")
        if bias_dir and bias_dir.exists():
            mlflow.log_artifacts(str(bias_dir), "bias")
        mlflow.log_dict(decision, "promotion_decision.json")
        mlflow.log_params({"checkpoint": str(checkpoint), "tau": candidate["tau"]})
        mlflow.log_metrics({f"test_{k}": v for k, v in candidate["test"].items()
                            if isinstance(v, float) and not math.isnan(v)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        version = mlflow.register_model(info.model_uri, name)
    tags = {"checkpoint": str(checkpoint), "tau": f"{candidate['tau']:.4f}",
            "test_recall_target_met": str(candidate["checks"]["urgent_recall"]["passed"]).lower(),
            "test_calibration_ok": str(candidate["checks"]["calibration_error"]["passed"]).lower(),
            "bias_result": bias["result"] if bias else "not run",
            "promotion": "promoted" if decision["promote"] else "rejected: " + "; ".join(decision["reasons"]),
            "warnings": "; ".join(decision["warnings"]) or "none"}
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, value)
    client.update_model_version(name, version.version, description=(
        f"{checkpoint.parent.name}: {decision['metric']} {decision['candidate_value']:.3f}, test "
        f"urgent recall {candidate['test']['urgent_recall']:.3f} (target {candidate['recall_target']}), "
        f"calibration error {candidate['test']['calibration_error']:.3f}. Warnings: {tags['warnings']}."))

    if decision["promote"]:
        if current:
            client.set_registered_model_alias(name, PREVIOUS, current.version)
        set_production(client, name, version.version)
    return {"name": name, "version": int(version.version), "run_id": run.info.run_id,
            "previous_production": int(current.version) if current else None, **decision}


def set_production(client: MlflowClient, name: str, version: str) -> None:
    client.set_registered_model_alias(name, PRODUCTION, version)
    with warnings.catch_warnings():               # stages are deprecated in MLflow 3
        warnings.simplefilter("ignore")
        try:
            client.transition_model_version_stage(name, version, "Production",
                                                  archive_existing_versions=True)
        except MlflowException:
            pass


def rollback(params: Dict[str, Any]) -> Dict[str, Any]:
    """Point Production back at the previous version (and remember the one rolled back)."""
    name = params["mlflow"]["registered_model_name"]
    client = MlflowClient()
    current = client.get_model_version_by_alias(name, PRODUCTION)
    try:
        previous = client.get_model_version_by_alias(name, PREVIOUS)
    except MlflowException:
        raise SystemExit(f"{name}: no previous version to roll back to")
    set_production(client, name, previous.version)
    client.set_registered_model_alias(name, PREVIOUS, current.version)
    client.set_model_version_tag(name, current.version, "rolled_back", "true")
    return {"name": name, "production": int(previous.version), "rolled_back": int(current.version)}


def status(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    name = params["mlflow"]["registered_model_name"]
    client = MlflowClient()
    rows = []
    for found in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        v = client.get_model_version(name, found.version)   # search results leave aliases empty
        aliases = list(v.aliases)
        rows.append({"version": int(v.version), "aliases": aliases, "stage": v.current_stage,
                     "promotion": v.tags.get("promotion"), "warnings": v.tags.get("warnings"),
                     "checkpoint": v.tags.get("checkpoint")})
    return rows


def push_to_gcs(params: Dict[str, Any], dry_run: bool = False, client=None,
                record: Optional[Path] = ROOT / "outputs" / "gcs_push.json") -> Dict[str, Any]:
    """Copy the Production version's model files to GCS under v<N>/, point latest.json at it and
    write the version number to version.txt (course Github_Labs/Lab4 pattern: versioned model in
    a bucket). Never overwrites a version."""
    import datetime
    import subprocess

    import mlflow.artifacts

    name = params["mlflow"]["registered_model_name"]
    registry_client = MlflowClient()
    version = registry_client.get_model_version_by_alias(name, PRODUCTION)
    bucket_name, _, prefix = params["registry"]["gcs_uri"].removeprefix("gs://").partition("/")
    root = prefix.rstrip("/")
    local = Path(mlflow.artifacts.download_artifacts(f"models:/{name}@{PRODUCTION}"))
    files = sorted(f for f in local.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    info = {"model": name, "version": int(version.version), "run_id": version.run_id,
            "tags": dict(version.tags), "git_commit": commit,
            "metrics": version_metrics(registry_client, version),
            "pushed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    if client is None and not dry_run:
        from google.cloud import storage
        client = storage.Client(project=params["registry"].get("gcp_project"))
    bucket = client.bucket(bucket_name) if client is not None else None
    # GCS numbering is its own (v1, v2, ...): version.txt holds the highest pushed number, so a
    # CI runner with a fresh MLflow registry (always version 1) never collides
    version_blob = bucket.blob(f"{root}/version.txt") if bucket is not None else None
    last = int(version_blob.download_as_text()) if version_blob is not None and version_blob.exists() else 0
    gcs_version = last + 1
    target = f"{root}/v{gcs_version}"
    info.update(gcs_version=gcs_version, gcs_path=f"gs://{bucket_name}/{target}/")
    result = {**info, "files": len(files), "bytes": sum(f.stat().st_size for f in files), "dry_run": dry_run}
    if dry_run:
        return result
    if any(True for _ in bucket.list_blobs(prefix=f"{target}/", max_results=1)):
        raise SystemExit(f"gs://{bucket_name}/{target}/ already exists; versions are never overwritten")
    for f in files:
        bucket.blob(f"{target}/{f.relative_to(local).as_posix()}").upload_from_filename(str(f))
    bucket.blob(f"{target}/model_info.json").upload_from_string(json.dumps(info, indent=2),
                                                                content_type="application/json")
    # latest.json = the deployed version (rollback-gcs moves it back);
    # version.txt = highest version number pushed (course Github_Labs/Lab4 version file)
    bucket.blob(f"{root}/latest.json").upload_from_string(json.dumps(info, indent=2),
                                                          content_type="application/json")
    version_blob.upload_from_string(str(gcs_version))
    if record is not None:          # the CI reads the pushed version number from here
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(result, indent=2))
    registry_client.set_model_version_tag(name, version.version, "gcs_path", info["gcs_path"])
    return result


def compare_deployed(params: Dict[str, Any], candidate_dir: Path, client=None) -> Dict[str, Any]:
    """Gate a new model against the one deployed in GCS (latest.json), for CI where the MLflow
    registry is fresh on every run. Same rules as gate(); if it fails, nothing is pushed and
    the deployed version stays (rollback by not promoting)."""
    candidate = json.loads((candidate_dir / "metrics.json").read_text())
    bucket_name, _, prefix = params["registry"]["gcs_uri"].removeprefix("gs://").partition("/")
    if client is None:
        from google.cloud import storage
        client = storage.Client(project=params["registry"].get("gcp_project"))
    blob = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/latest.json")
    deployed = json.loads(blob.download_as_text()) if blob.exists() else None
    decision = gate(candidate, deployed["metrics"] if deployed and "metrics" in deployed else None,
                    params["registry"])
    decision["deployed_version"] = deployed["version"] if deployed else None
    return decision


def rollback_gcs(params: Dict[str, Any], to: Optional[int] = None, client=None) -> Dict[str, Any]:
    """Point latest.json back at an earlier pushed version (default: the one before the current).
    The serving image is rolled back the same way by re-tagging, e.g.
    gcloud artifacts docker tags add IMAGE:v<N> IMAGE:latest"""
    bucket_name, _, prefix = params["registry"]["gcs_uri"].removeprefix("gs://").partition("/")
    root = prefix.rstrip("/")
    if client is None:
        from google.cloud import storage
        client = storage.Client(project=params["registry"].get("gcp_project"))
    bucket = client.bucket(bucket_name)
    current = json.loads(bucket.blob(f"{root}/latest.json").download_as_text())
    target = to if to is not None else current["gcs_version"] - 1
    blob = bucket.blob(f"{root}/v{target}/model_info.json")
    if target < 1 or not blob.exists():
        raise SystemExit(f"no pushed version v{target} to roll back to")
    info = json.loads(blob.download_as_text())
    info.setdefault("gcs_version", target)
    info["rolled_back_from"] = current["gcs_version"]
    bucket.blob(f"{root}/latest.json").upload_from_string(json.dumps(info, indent=2),
                                                          content_type="application/json")
    return {"deployed": target, "rolled_back_from": current["gcs_version"]}


def export(params: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """Copy the Production model to a local folder for the serving image (no network)."""
    import shutil

    import mlflow.artifacts

    name = params["mlflow"]["registered_model_name"]
    version = MlflowClient().get_model_version_by_alias(name, PRODUCTION)
    if out.exists():
        shutil.rmtree(out)
    mlflow.artifacts.download_artifacts(f"models:/{name}@{PRODUCTION}", dst_path=str(out))
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache)
    info = {"model": name, "version": int(version.version), "run_id": version.run_id,
            "tau": version.tags.get("tau"), "warnings": version.tags.get("warnings")}
    (out / "model_info.json").write_text(json.dumps(info, indent=2))
    return info


def example_input(params: Dict[str, Any], n: int = 2) -> pd.DataFrame:
    """A couple of pad_test photos with their answers, as the serving input example."""
    from src.training.dataset import load_table, resolve
    df = load_table(params, [params["data"]["splits"]["pad_test"]]).head(n)
    images = resolve(params["data"]["images_dir"])
    out = pd.DataFrame({"image_path": [str(images / f"{i}.jpg") for i in df["image_id"]]})
    for c in params["data"]["answer_columns"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SkinSort model registry")
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="register a model and promote it if it passes the gate")
    reg.add_argument("--from-selection", type=Path,
                     help="outputs/selection/model_selection.json from select_model.py (sets the next three)")
    reg.add_argument("--checkpoint", type=Path, help="fusion checkpoint from fit_final.py")
    reg.add_argument("--evaluation", type=Path, help="outputs/evaluation/<name>")
    reg.add_argument("--bias", type=Path, help="outputs/bias/<name>")
    reg.add_argument("--fail-if-rejected", action="store_true", help="exit 1 if not promoted (for CI)")
    sub.add_parser("status", help="list versions, aliases and promotion results")
    sub.add_parser("rollback", help="move Production back to the previous version")
    rbg = sub.add_parser("rollback-gcs", help="point GCS latest.json back at an earlier version")
    rbg.add_argument("--to", type=int, help="version number (default: the one before the deployed)")
    cmp = sub.add_parser("compare-deployed", help="gate a new model against the GCS latest.json (CI)")
    cmp.add_argument("--evaluation", type=Path, required=True, help="outputs/evaluation/<name>")
    exp = sub.add_parser("export", help="copy the Production model to a local folder (for Docker serving)")
    exp.add_argument("--out", type=Path, default=ROOT / "models" / "triage_model")
    push = sub.add_parser("push", help="copy the Production version to GCS (registry.gcs_uri)")
    push.add_argument("--dry-run", action="store_true", help="only show what would be uploaded")
    args = parser.parse_args()

    params = load_params()
    setup_mlflow(params)
    if args.command == "register":
        if args.from_selection:
            chosen = json.loads(args.from_selection.read_text())
            if not chosen.get("selected"):
                sys.exit(f"{args.from_selection}: no model was selected")
            args.checkpoint, args.evaluation = Path(chosen["checkpoint"]), Path(chosen["eval_dir"])
            bias_dir = ROOT / "outputs" / "bias" / chosen["selected"]
            args.bias = args.bias or (bias_dir if bias_dir.exists() else None)
        if not (args.checkpoint and args.evaluation):
            sys.exit("register needs --from-selection or both --checkpoint and --evaluation")
        r = register(params, args.checkpoint.resolve(), args.evaluation.resolve(),
                     args.bias.resolve() if args.bias else None, example_input(params))
        verdict = "PROMOTED to production" if r["promote"] else "REJECTED: " + "; ".join(r["reasons"])
        print(f"{r['name']} version {r['version']}: {verdict}")
        print(f"  {r['metric']} {r['candidate_value']:.3f}"
              + (f" (Production was {r['production_value']:.3f}, v{r['previous_production']})"
                 if r["production_value"] is not None else " (no Production version before)"))
        if r["warnings"]:
            print("  warnings:", "; ".join(r["warnings"]))
        if args.fail_if_rejected and not r["promote"]:
            sys.exit(1)
    elif args.command == "rollback":
        r = rollback(params)
        print(f"{r['name']}: production -> v{r['production']} (rolled back v{r['rolled_back']})")
    elif args.command == "rollback-gcs":
        r = rollback_gcs(params, args.to)
        print(f"GCS latest.json -> v{r['deployed']} (was v{r['rolled_back_from']})")
    elif args.command == "compare-deployed":
        d = compare_deployed(params, args.evaluation.resolve())
        verdict = "OK to deploy" if d["promote"] else "REJECTED: " + "; ".join(d["reasons"])
        print(f"new model vs deployed v{d['deployed_version']}: {verdict}")
        sys.exit(0 if d["promote"] else 1)
    elif args.command == "export":
        info = export(params, args.out.resolve())
        print(f"{info['model']} v{info['version']} exported to {args.out}")
    elif args.command == "push":
        r = push_to_gcs(params, dry_run=args.dry_run)
        action = "would upload" if r["dry_run"] else "uploaded"
        print(f"{r['model']} (registry v{r['version']}) as GCS v{r['gcs_version']}: {action} {r['files']} files "
              f"({r['bytes'] / 1e6:.1f} MB) to {r['gcs_path']}")
    else:
        for row in status(params):
            print(row)


if __name__ == "__main__":
    main()
