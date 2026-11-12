"""Copy the key charts and reports from outputs/ (gitignored) into docs/figures/ (committed),
so the results can be seen on GitHub without re-running anything (assignment §4).

    python -m src.training.collect_figures

Writes docs/figures/<group>/... and docs/figures/README.md listing every file and its source.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from src.training.dataset import ROOT

OUT = ROOT / "outputs"
DEST = ROOT / "docs" / "figures"

FIGURES: Dict[str, List[Tuple[str, str]]] = {
    "selection": [
        ("selection/backbone_comparison.png", "Backbones: validation urgent AUC, size and CPU speed"),
        ("selection/phone_comparison.png", "Phone candidates and baselines (selection metric)"),
        ("selection/model_selection.md", "Selection tables and the reason for the choice"),
    ],
    "validation": [
        ("evaluation/image_main/confusion_matrix.png", "Image model, main test set"),
        ("evaluation/image_main/calibration.png", "Image model calibration, main test set"),
        ("evaluation/{model}/confusion_matrix.png", "Selected phone model, pad_test"),
        ("evaluation/{model}/calibration.png", "Phone model calibration, pad_test"),
    ],
    "bias": [
        ("bias/{model}/val_sex.png", "Phone model, urgent recall by sex (out-of-fold)"),
        ("bias/{model}/val_body_site.png", "Phone model, by body site"),
        ("bias/{model}/val_skin_tone.png", "Phone model, by skin tone"),
        ("bias/{model}/val_age_group.png", "Phone model, by age group"),
        ("bias/{model}/bias_report.md", "Phone model bias report"),
        ("bias/{model}/mitigation.md", "Phone model mitigation (per-group thresholds)"),
        ("bias/image_main/test_age_group.png", "Image model, by age group (test)"),
        ("bias/image_main/test_body_site.png", "Image model, by body site (test)"),
        ("bias/image_main/test_source.png", "Image model, by source (test)"),
        ("bias/image_main/fairness_skin_tone.png", "Image model, by skin tone (fairness_test)"),
        ("bias/image_main/bias_report.md", "Image model bias report"),
        ("bias/image_main/mitigation.md", "Image model mitigation"),
    ],
    "tuning": [
        ("tuning_b3/sensitivity.png", "Hyperopt trials on the final backbone: CV AUC vs each hyperparameter"),
        ("tuning_b3/search_progress.png", "Hyperopt search progress (final backbone)"),
        ("tuning_b3/lr_sweep.png", "Learning rate, one at a time"),
        ("tuning/sensitivity.png", "First search, on the leaky fine-tuned features (kept for the write-up)"),
    ],
    "sensitivity": [
        ("explain/{model}/shap_importance.png", "SHAP: what moves P(urgent)"),
        ("explain/{model}/shap_answers.png", "SHAP per answer (yes vs no)"),
        ("analysis/robustness.png", "Robustness to blur, light, compression, resolution"),
        ("analysis/thresholds.png", "Recall vs flagged share for every threshold"),
    ],
}


def selected_model(out: Path = OUT) -> str:
    sel = out / "selection" / "model_selection.json"
    return (json.loads(sel.read_text()).get("selected") if sel.exists() else None) or "fusion_oof"


def collect(out: Path = OUT, dest: Path = DEST) -> List[str]:
    """Copy what exists; return the list of sources that were missing. "{model}" in a path is
    the model chosen by select_model.py."""
    missing = []
    model = selected_model(out)
    lines = ["# Figures", "",
             "Copied from `outputs/` by `python -m src.training.collect_figures`. "
             "Discussed in [../model_pipeline_report.md](../model_pipeline_report.md).", ""]
    for group, items in FIGURES.items():
        lines += [f"## {group}", "", "| File | What it shows | Source |", "|---|---|---|"]
        for src, caption in items:
            src = src.format(model=model)
            path = out / src
            if not path.exists():
                missing.append(src)
                continue
            target = dest / group / src.replace("/", "_")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            rel = target.relative_to(dest).as_posix()
            lines.append(f"| [{rel}]({rel}) | {caption} | `outputs/{src}` |")
        lines.append("")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "README.md").write_text("\n".join(lines))
    return missing


def main() -> None:
    missing = collect()
    print(f"figures copied to {DEST}" + (f"; missing: {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
