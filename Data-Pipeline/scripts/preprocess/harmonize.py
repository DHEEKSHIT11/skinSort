"""Put all four datasets into one common schema and attach the urgency tier.

Input : data/raw/<dataset>/metadata.csv (+ images/)
Output: data/interim/harmonized.csv, one row per image, columns COMMON_COLUMNS.

Rows are dropped (and counted) when their diagnosis cannot be mapped or their
image file is missing, e.g. Fitzpatrick17k rows whose link was dead.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from scripts.common.config import load_params, load_yaml, resolve_path
from scripts.common.logging_utils import configure_console_logging, get_logger

logger = get_logger("preprocess.harmonize")

ANSWER_COLUMNS = ["q_itch", "q_grew", "q_hurt", "q_changed", "q_bleed", "q_elevation"]
COMMON_COLUMNS = [
    "image_id", "source", "source_image_id", "image_path", "image_type",
    "patient_id", "lesion_id", "diagnosis_raw", "diagnosis", "urgency",
    "age", "sex", "body_site", "fitzpatrick", *ANSWER_COLUMNS, "license",
]


class HarmonizationError(Exception):
    """Raised when the mapping configs are inconsistent with the data."""


# --- small, reusable mapping helpers -----------------------------------------

def apply_rules(df: pd.DataFrame, rules: list) -> pd.Series:
    """First-match-wins substring rules -> canonical diagnosis (NaN if none match)."""
    result = pd.Series(np.nan, index=df.index, dtype=object)
    for rule in rules:
        if rule["field"] not in df.columns:
            continue
        values = df[rule["field"]].astype("string").str.lower()
        hit = values.str.contains(rule["contains"].lower(), regex=False).fillna(False)
        result = result.mask(result.isna() & hit.astype(bool), rule["diagnosis"])
    return result


def map_body_site(values: pd.Series, site_map: Dict[str, list]) -> pd.Series:
    """Map free-text body sites to the coarse shared set; unknown/missing -> 'unknown'."""
    lookup = {alias.lower(): site for site, aliases in site_map.items() for alias in aliases}
    return values.astype("string").str.strip().str.lower().map(lookup).fillna("unknown")


def normalise_sex(values: pd.Series) -> pd.Series:
    sex = values.astype("string").str.strip().str.lower()
    return sex.where(sex.isin(["male", "female"]), "unknown").fillna("unknown")


def clean_age(values: pd.Series) -> pd.Series:
    """Numeric age; impossible values (<0 or >120) become missing."""
    age = pd.to_numeric(values, errors="coerce")
    return age.where(age.between(0, 120))


def parse_answer(values: pd.Series) -> pd.Series:
    """PAD-UFES-20 answers 'True'/'False'/'UNK' -> 1.0 / 0.0 / NaN (missing)."""
    text = values.astype("string").str.strip().str.lower()
    return text.map({"true": 1.0, "false": 0.0}).astype(float)


def attach_urgency(df: pd.DataFrame, urgency_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add the urgency tier; fail loudly if a canonical diagnosis has no tier."""
    mapping = urgency_cfg["mapping"]
    unknown = set(df["diagnosis"].dropna()) - set(mapping)
    if unknown:
        raise HarmonizationError(
            f"Diagnoses missing from urgency_mapping.yaml: {sorted(unknown)}"
        )
    bad_tiers = set(mapping.values()) - set(urgency_cfg["tiers"])
    if bad_tiers:
        raise HarmonizationError(f"Unknown urgency tiers in mapping: {sorted(bad_tiers)}")
    return df.assign(urgency=df["diagnosis"].map(mapping))


# --- one function per dataset -------------------------------------------------

def _col(meta: pd.DataFrame, name: str) -> pd.Series:
    """Column if present, otherwise an all-missing column (sources differ in fields)."""
    return meta[name] if name in meta.columns else pd.Series(np.nan, index=meta.index, dtype=object)


def _frame(source: str, ids: pd.Series, **columns) -> pd.DataFrame:
    out = pd.DataFrame({"source": source, "source_image_id": ids.astype(str)}, index=ids.index)
    for name, value in columns.items():
        out[name] = value
    out["image_id"] = source + "__" + out["source_image_id"]
    return out


def harmonize_isic(meta: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    # anatom_site_2 is more specific (hand / foot); fall back to anatom_site_1.
    site = map_body_site(_col(meta, "anatom_site_2"), cfg["body_site"])
    coarse = map_body_site(_col(meta, "anatom_site_1"), cfg["body_site"])
    site = site.mask(site == "unknown", coarse)
    diagnosis_raw = pd.concat([_col(meta, f"diagnosis_{i}") for i in (1, 2, 3)], axis=1) \
        .astype("string").fillna("").agg(" | ".join, axis=1)
    return _frame(
        "isic", meta["isic_id"],
        image_path="isic/images/" + meta["isic_id"] + ".jpg",
        image_type="dermoscopic",
        patient_id=_col(meta, "patient_id"),
        lesion_id=_col(meta, "lesion_id"),
        diagnosis_raw=diagnosis_raw,
        diagnosis=apply_rules(meta, cfg["diagnosis"]["isic"]["rules"]),
        age=clean_age(_col(meta, "age_approx")),
        sex=normalise_sex(_col(meta, "sex")),
        body_site=site,
        fitzpatrick=_col(meta, "fitzpatrick_skin_type").map(cfg["fitzpatrick_roman"]),
        license=_col(meta, "copyright_license"),
    )


def harmonize_ham10000(meta: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    return _frame(
        "ham10000", meta["image_id"],
        image_path="ham10000/images/" + meta["image_id"] + ".jpg",
        image_type="dermoscopic",
        patient_id=np.nan,
        lesion_id=meta["lesion_id"],
        diagnosis_raw=meta["dx"],
        diagnosis=meta["dx"].map(cfg["diagnosis"]["ham10000"]["codes"]),
        age=clean_age(meta["age"]),
        sex=normalise_sex(meta["sex"]),
        body_site=map_body_site(meta["localization"], cfg["body_site"]),
        fitzpatrick=np.nan,
        license="CC BY-NC 4.0",
    )


def harmonize_pad_ufes(meta: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    answers = {f"q_{c}": parse_answer(meta[c])
               for c in ["itch", "grew", "hurt", "changed", "bleed", "elevation"]}
    fitz = pd.to_numeric(meta["fitspatrick"], errors="coerce")  # sic: column name in source
    return _frame(
        "pad_ufes_20", meta["img_id"].str.replace(r"\.png$", "", regex=True),
        image_path="pad_ufes_20/images/" + meta["img_id"],
        image_type="smartphone",
        patient_id=meta["patient_id"],
        lesion_id=meta["patient_id"] + "_" + meta["lesion_id"].astype(str),
        diagnosis_raw=meta["diagnostic"],
        diagnosis=meta["diagnostic"].map(cfg["diagnosis"]["pad_ufes_20"]["codes"]),
        age=clean_age(meta["age"]),
        sex=normalise_sex(meta["gender"]),
        body_site=map_body_site(meta["region"], cfg["body_site"]),
        fitzpatrick=fitz.where(fitz.between(1, 6)),
        license="CC BY 4.0",
        **answers,
    )


def harmonize_fitzpatrick17k(meta: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    meta = meta.dropna(subset=["md5hash"]).drop_duplicates(subset="md5hash")
    fitz = pd.to_numeric(meta["fitzpatrick_scale"], errors="coerce")
    return _frame(
        "fitzpatrick17k", meta["md5hash"],
        image_path="fitzpatrick17k/images/" + meta["md5hash"] + ".jpg",
        image_type="clinical",
        patient_id=np.nan,
        lesion_id=np.nan,
        diagnosis_raw=meta["label"],
        diagnosis=apply_rules(meta, cfg["diagnosis"]["fitzpatrick17k"]["rules"]),
        age=np.nan,
        sex="unknown",
        body_site="unknown",
        fitzpatrick=fitz.where(fitz.between(1, 6)),  # -1 means "unknown" in the source
        license="non-commercial research",
    )


HARMONIZERS: Dict[str, Callable[[pd.DataFrame, Dict], pd.DataFrame]] = {
    "isic": harmonize_isic,
    "ham10000": harmonize_ham10000,
    "pad_ufes_20": harmonize_pad_ufes,
    "fitzpatrick17k": harmonize_fitzpatrick17k,
}


# --- pipeline step ----------------------------------------------------------

def harmonize_all(raw_dir: Path, harm_cfg: Dict, urgency_cfg: Dict,
                  datasets: Optional[list] = None):
    """Harmonize every dataset found in raw_dir. Returns (dataframe, report)."""
    frames, report = [], {}
    for source in datasets or HARMONIZERS:
        meta_path = raw_dir / source / "metadata.csv"
        if not meta_path.is_file():
            raise HarmonizationError(f"Missing {meta_path}; run the ingest step first")
        frame = HARMONIZERS[source](pd.read_csv(meta_path, low_memory=False), harm_cfg)
        unmapped = frame["diagnosis"].isna()
        has_image = frame["image_path"].map(lambda p: (raw_dir / p).is_file())
        report[source] = {
            "rows": int(len(frame)),
            "unmapped_diagnosis": int(unmapped.sum()),
            "missing_image": int((~has_image & ~unmapped).sum()),
        }
        if unmapped.any():
            logger.warning("%s: %d rows with unmapped diagnosis, e.g. %s", source,
                           unmapped.sum(), frame.loc[unmapped, "diagnosis_raw"].head(3).tolist())
        frame = frame[~unmapped & has_image]
        report[source]["kept"] = int(len(frame))
        logger.info("%s: %s", source, report[source])
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = attach_urgency(df, urgency_cfg)
    for column in COMMON_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[COMMON_COLUMNS], report


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = resolve_path(params, "interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    df, report = harmonize_all(
        resolve_path(params, "raw"),
        load_yaml(resolve_path(params, "harmonization")),
        load_yaml(resolve_path(params, "urgency_mapping")),
    )
    df.to_csv(out_dir / "harmonized.csv", index=False)
    report["total"] = {
        "images": int(len(df)),
        "by_urgency": df["urgency"].value_counts().to_dict(),
        "by_source": df["source"].value_counts().to_dict(),
    }
    logs_dir = resolve_path(params, "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "harmonize_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Harmonized %d images: %s", len(df), report["total"])
    return report["total"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize datasets into one schema")
    parser.add_argument("--params", type=Path, default=None)
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
