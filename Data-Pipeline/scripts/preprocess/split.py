"""Split into train / val / test by group (patient or lesion), never by photo.

* Dermoscopic pool (ISIC + HAM10000): train / val / test
* PAD-UFES-20 (smartphone): pad_train / pad_val / pad_test
  (pad_test is the "real phone photos" test set from the scoping doc)
* Fitzpatrick17k: fairness_test only, never trained on

Splits are stratified by urgency: groups are shuffled (seeded) within each
urgency tier and handed out until each split holds its share of images.
A final check fails the step if any group appears in two splits.

Input : deduped.csv + features.csv
Output: data/processed/dataset.csv (metadata + features + split),
        logs/split_report.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from scripts.common.config import load_params, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger

logger = get_logger("preprocess.split")

URGENCY_ORDER = {"urgent": 0, "priority": 1, "routine": 2}


class LeakageError(Exception):
    """Raised when the same group (patient/lesion/duplicate) is in two splits."""


def assign_splits(df: pd.DataFrame, ratios: Dict[str, float], seed: int,
                  prefix: str = "") -> pd.Series:
    """Group-aware, urgency-stratified split. Returns the split name per row."""
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {ratios}")
    # A group's tier is its most urgent image, so rare urgent cases are balanced.
    groups = df.groupby("group_id").agg(
        size=("image_id", "size"),
        tier=("urgency", lambda s: min(s.map(URGENCY_ORDER))),
    )
    rng = np.random.default_rng(seed)
    names = list(ratios)
    group_split = {}
    for _, tier_groups in groups.groupby("tier"):
        order = rng.permutation(tier_groups.index.to_numpy())
        total = tier_groups["size"].sum()
        targets = np.cumsum([ratios[n] * total for n in names])
        filled = 0
        for group in order:
            # hand the group to the first split that is still below its cumulative target
            idx = int(np.searchsorted(targets, filled, side="right"))
            group_split[group] = prefix + names[min(idx, len(names) - 1)]
            filled += tier_groups.at[group, "size"]
    return df["group_id"].map(group_split)


def check_no_leakage(df: pd.DataFrame) -> None:
    """Fail if any group or perceptual hash appears in more than one split."""
    for key in ("group_id", "phash"):
        spread = df.groupby(key)["split"].nunique()
        leaked = spread[spread > 1]
        if len(leaked):
            raise LeakageError(
                f"{len(leaked)} {key} values appear in more than one split, "
                f"e.g. {leaked.index[:3].tolist()}"
            )


def build_splits(df: pd.DataFrame, ratios: Dict[str, float], seed: int) -> pd.Series:
    split = pd.Series(index=df.index, dtype=object)
    derm = df["source"].isin(["isic", "ham10000"])
    pad = df["source"] == "pad_ufes_20"
    fitz = df["source"] == "fitzpatrick17k"
    if derm.any():
        split[derm] = assign_splits(df[derm], ratios, seed)
    if pad.any():
        split[pad] = assign_splits(df[pad], ratios, seed, prefix="pad_")
    split[fitz] = "fairness_test"
    return split


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    interim, processed = resolve_path(params, "interim"), resolve_path(params, "processed")
    processed.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(interim / "deduped.csv", low_memory=False)
    features = pd.read_csv(interim / "features.csv", low_memory=False)
    df["split"] = build_splits(df, params["preprocess"]["split"]["ratios"], params["seed"])
    check_no_leakage(df)

    dataset = df.merge(features, on="image_id", how="left", suffixes=("", "_feat"),
                       validate="one_to_one")
    dataset["processed_image"] = "images/" + dataset["image_id"] + ".jpg"
    dataset.to_csv(processed / "dataset.csv", index=False)

    report = {
        split: {"images": int(len(g)), "groups": int(g["group_id"].nunique()),
                "urgency": g["urgency"].value_counts().to_dict()}
        for split, g in dataset.groupby("split")
    }
    logs_dir = resolve_path(params, "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "split_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Splits: %s", {k: v["images"] for k, v in report.items()})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-aware train/val/test split")
    parser.add_argument("--params", type=Path, default=None)
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
