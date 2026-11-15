"""Run the model pipeline end to end (Docker trainer service and CI use this).

Steps, each a module that also runs on its own:
  1. fit_final     phone fusion head on the selected backbone (optionally reweighted by group),
                   out-of-fold tau + calibration
  2. bias          slice-based bias check on the out-of-fold rows (+ pad_test)
  3. select_model  candidates + baselines, search_runs ranking, bias consideration
  4. registry      register the selected model; the gate decides if it becomes Production
                   (a worse model is rejected, so Production stays on the previous version)
  5. export        copy Production to models/triage_model for the serving image
  6. figures       copy the key charts and reports into docs/figures/

The image backbones are trained separately on a GPU (src.training.train on the GCP VM);
this pipeline needs their checkpoint (outputs/models/<backbone>/best.pt) and the DVC data.

    python -m src.training.pipeline                       # everything
    python -m src.training.pipeline --steps fit_final bias # some steps
    python -m src.training.pipeline --fail-if-rejected     # exit 1 if the gate rejects (CI)

Writes outputs/pipeline_summary.json (read by src/notify.py for the email).
"""
import argparse
import json
import subprocess
import sys
import time
from typing import Dict, List

from src.training.dataset import ROOT, load_params

STEPS = ["fit_final", "bias", "select_model", "registry", "export", "figures"]


def commands(params: Dict, backbone: str, name: str, trained_on: List[str], fail_if_rejected: bool,
             reweight: List[str] = (), mitigate: List[str] = ()) -> Dict[str, List[str]]:
    py = [sys.executable, "-m"]
    register = py + ["src.training.registry", "register",
                     "--from-selection", str(ROOT / "outputs" / "selection" / "model_selection.json")]
    if fail_if_rejected:
        register.append("--fail-if-rejected")
    fit = py + ["src.training.fit_final", "--image-checkpoint",
                str(ROOT / "outputs" / "models" / backbone / "best.pt"),
                "--trained-on", *trained_on, "--settings", "params", "--name", name]
    bias = py + ["src.training.bias", "--name", name, "--include-val"]
    return {
        "fit_final": fit + (["--reweight", *reweight] if reweight else []),
        "bias": bias + (["--mitigate", *mitigate] if mitigate else []),
        "select_model": py + ["src.training.select_model", "--no-speed"],
        "registry": register,
        "export": py + ["src.training.registry", "export"],
        "figures": py + ["src.training.collect_figures"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SkinSort model pipeline")
    parser.add_argument("--steps", nargs="*", default=STEPS, choices=STEPS)
    parser.add_argument("--backbone", help="default: params.yaml pipeline.backbone")
    parser.add_argument("--trained-on", nargs="*", default=["train"],
                        help="splits the backbone learned from (for checkpoints that do not record it)")
    parser.add_argument("--name", help="name of the new phone model (default: params.yaml pipeline.model_name)")
    parser.add_argument("--fail-if-rejected", action="store_true")
    args = parser.parse_args()

    params = load_params()
    cfg = params["pipeline"]
    cmds = commands(params, args.backbone or cfg["backbone"], args.name or cfg["model_name"], args.trained_on,
                    args.fail_if_rejected, cfg.get("reweight", []), cfg.get("mitigate", []))
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": [], "status": "success"}
    out = ROOT / "outputs" / "pipeline_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        print(f"\n=== {step}: {' '.join(cmds[step][2:])}", flush=True)
        start = time.time()
        code = subprocess.run(cmds[step], cwd=ROOT).returncode
        summary["steps"].append({"step": step, "exit_code": code, "seconds": round(time.time() - start)})
        if code != 0:
            summary["status"] = f"failed at {step}"
            break
    selection = ROOT / "outputs" / "selection" / "model_selection.json"
    if selection.exists():
        s = json.loads(selection.read_text())
        summary["selected"] = {"model": s.get("selected"), "bias_result": s.get("bias_result"),
                               "reason": s.get("reason")}
    out.write_text(json.dumps(summary, indent=2))
    print(f"\npipeline {summary['status']}; summary in {out}")
    sys.exit(0 if summary["status"] == "success" else 1)


if __name__ == "__main__":
    main()
