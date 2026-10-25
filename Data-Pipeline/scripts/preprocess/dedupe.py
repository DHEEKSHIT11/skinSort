"""Remove duplicate images and build leakage-safe groups for splitting.

Many HAM10000 photos are also in the ISIC Archive. If the same photo landed
in both train and test, the model would look better than it is. Two passes:

1. Copies that cross datasets: the same ISIC ID, an identical perceptual hash
   in two datasets, or a near-copy between two *pools* (dermoscopic training
   pool, PAD-UFES-20, Fitzpatrick17k). One copy is kept, chosen by
   ``source_priority``. Pools must not share images because each has its own
   test set.
2. Near-copies inside a pool (perceptual hash within ``max_hamming_distance``
   bits): kept, but their groups are merged so they land in the same split.

Every image gets a ``group_id``: patient if known, else lesion, else the image
itself, merged across near-copies. The split step never divides a group.

Input : quality_passed.csv
Output: data/interim/deduped.csv, duplicates_dropped.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from scripts.common.config import load_params, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger

logger = get_logger("preprocess.dedupe")

HASH_BITS = 64
# Images in different pools end up in different test sets, so they must not overlap.
POOLS = {"isic": "dermoscopic", "ham10000": "dermoscopic",
         "pad_ufes_20": "pad_ufes_20", "fitzpatrick17k": "fitzpatrick17k"}


class UnionFind:
    """Minimal union-find used to merge groups connected by near-duplicates."""

    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # keep the lexicographically smaller root so results are deterministic
            if root_b < root_a:
                root_a, root_b = root_b, root_a
            self.parent[root_b] = root_a


def base_group(df: pd.DataFrame) -> pd.Series:
    """Patient ID if known, else lesion ID, else the image itself (prefixed by source)."""
    key = df["patient_id"].where(df["patient_id"].notna(), df["lesion_id"])
    key = key.where(key.notna(), df["image_id"])
    return df["source"] + "/" + key.astype(str)


def near_duplicate_pairs(hashes: pd.Series, max_distance: int) -> List[Tuple[int, int, int]]:
    """Find all pairs of 64-bit hashes within max_distance bits.

    Pigeonhole trick: split each hash into (max_distance + 1) bands. Two hashes
    that differ in at most max_distance bits must agree exactly on at least one
    band, so we only compare hashes that share a band. Returns (i, j, distance)
    with positional indices i < j.
    """
    values = [int(h, 16) for h in hashes]
    n_bands = max_distance + 1
    edges = [round(b * HASH_BITS / n_bands) for b in range(n_bands + 1)]
    candidates = set()
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (1 << (hi - lo)) - 1
        buckets = defaultdict(list)
        for idx, value in enumerate(values):
            buckets[(value >> lo) & mask].append(idx)
        for members in buckets.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    candidates.add((members[a], members[b]))
    pairs = []
    for i, j in candidates:
        distance = (values[i] ^ values[j]).bit_count()
        if distance <= max_distance:
            pairs.append((i, j, distance))
    return sorted(pairs)


def dedupe(df: pd.DataFrame, cfg: Dict[str, Any]):
    """Return (kept dataframe with group_id, dropped dataframe with reason)."""
    priority = {source: rank for rank, source in enumerate(cfg["source_priority"])}
    df = df.assign(_rank=df["source"].map(priority).fillna(len(priority)))
    df = df.sort_values(["_rank", "image_id"]).reset_index(drop=True)

    # Pass 1a: same original ISIC ID in two datasets (HAM10000 image_id == ISIC ID).
    same_id = df.duplicated(subset="source_image_id", keep="first") & \
        df["source"].isin(["isic", "ham10000"])
    dropped = [df[same_id].assign(drop_reason="same_isic_id")]
    df = df[~same_id].reset_index(drop=True)

    # Pass 1b + 2 use perceptual hashes.
    pairs = near_duplicate_pairs(df["phash"], cfg["max_hamming_distance"])
    source, pool = df["source"].to_numpy(), df["source"].map(POOLS).to_numpy()
    # Rows are sorted by source priority, so j (the later index) is the copy to drop.
    cross = {}
    for i, j, d in pairs:
        if pool[i] != pool[j]:
            cross.setdefault(j, "near_copy_other_pool")
        elif d == 0 and source[i] != source[j]:
            cross.setdefault(j, "identical_image_other_dataset")
    drop_idx = sorted(cross)
    dropped.append(df.loc[drop_idx].assign(drop_reason=[cross[j] for j in drop_idx]))

    groups = base_group(df)
    uf = UnionFind(groups.unique())
    merges = 0
    for i, j, _ in pairs:
        if i in cross or j in cross:
            continue
        if uf.find(groups[i]) != uf.find(groups[j]):
            merges += 1
        uf.union(groups[i], groups[j])
    df["group_id"] = groups.map(uf.find)
    kept = df.drop(index=drop_idx).drop(columns="_rank").reset_index(drop=True)

    dropped_df = pd.concat(dropped, ignore_index=True).drop(columns="_rank")
    logger.info("Near-duplicate pairs: %d; groups merged: %d", len(pairs), merges)
    return kept, dropped_df, {"near_duplicate_pairs": len(pairs), "groups_merged": merges}


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    interim = resolve_path(params, "interim")
    interim.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(interim / "quality_passed.csv", low_memory=False)
    kept, dropped, info = dedupe(df, params["preprocess"]["dedupe"])
    kept.to_csv(interim / "deduped.csv", index=False)
    dropped.to_csv(interim / "duplicates_dropped.csv", index=False)
    summary = {
        "kept": int(len(kept)),
        "dropped": dropped["drop_reason"].value_counts().to_dict(),
        "groups": int(kept["group_id"].nunique()),
        **info,
    }
    logger.info("Dedupe: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove duplicates and build split groups")
    parser.add_argument("--params", type=Path, default=None)
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
