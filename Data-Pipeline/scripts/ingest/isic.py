"""Download a capped, diagnosis-stratified subset of the ISIC Archive (API v2).

Output (data/raw/isic/):
    metadata.csv   one row per image: isic_id, stratum, license, clinical metadata
    images/        ISIC_XXXXXXX.jpg
    manifest.csv   checksum of every downloaded image
    failures.csv   images that could not be downloaded (if any)
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.common.config import load_params, raw_dataset_dir
from scripts.common.files import read_manifest, write_manifest
from scripts.common.http_utils import DownloadError, build_session, download_file, get_json
from scripts.common.logging_utils import configure_console_logging, get_logger

DATASET = "isic"
logger = get_logger(f"ingest.{DATASET}")


def flatten_record(record: Dict[str, Any], stratum: str, variant: str) -> Dict[str, Any]:
    """Turn one nested API result into a flat metadata row."""
    metadata = record.get("metadata", {})
    image_file = record.get("files", {}).get(variant, {})
    row = {
        "isic_id": record["isic_id"],
        "stratum": stratum,
        "copyright_license": record.get("copyright_license"),
        "attribution": record.get("attribution"),
        "image_url": image_file.get("url"),
        "image_bytes": image_file.get("size"),
    }
    for section in ("acquisition", "clinical"):
        row.update(metadata.get(section, {}))
    return row


def fetch_stratum(session, api_url: str, stratum: Dict, page_size: int, variant: str) -> List[Dict]:
    """Page through the search API with its cursor until max_images rows are collected."""
    rows: List[Dict] = []
    url: Optional[str] = api_url
    query_params: Optional[Dict] = {"query": stratum["query"], "limit": page_size}
    while url and len(rows) < stratum["max_images"]:
        page = get_json(session, url, params=query_params)
        for record in page.get("results", []):
            rows.append(flatten_record(record, stratum["name"], variant))
            if len(rows) >= stratum["max_images"]:
                break
        url = page.get("next")
        query_params = None  # the "next" URL already carries the query and cursor
    logger.info("Stratum %-25s %5d images listed", stratum["name"], len(rows))
    return rows


def build_metadata(session, cfg: Dict[str, Any]) -> pd.DataFrame:
    """List all strata and drop images that matched more than one stratum."""
    rows = []
    for stratum in cfg["strata"]:
        rows.extend(fetch_stratum(session, cfg["api_url"], stratum, cfg["page_size"],
                                  cfg["image_variant"]))
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise DownloadError("ISIC search returned no images; check the strata queries")
    before = len(metadata)
    metadata = metadata.drop_duplicates(subset="isic_id", keep="first")
    if len(metadata) < before:
        logger.info("Dropped %d images listed in more than one stratum", before - len(metadata))
    return metadata.sort_values("isic_id").reset_index(drop=True)


def download_images(session, metadata: pd.DataFrame, images_dir: Path,
                    done: Dict[str, Dict], max_workers: int):
    """Download every image in parallel; returns (records, failures)."""
    records, failures, todo = [], [], []
    for row in metadata.itertuples(index=False):
        name = f"{row.isic_id}.jpg"
        if name in done and (images_dir / name).exists():
            records.append(done[name])
        elif not row.image_url:
            failures.append({"isic_id": row.isic_id, "error": "no image URL in API response"})
        else:
            todo.append(row)
    logger.info("%d images already present, %d to download", len(records), len(todo))

    # The API's reported file size does not match the served file (ISIC embeds
    # extra metadata in the JPEG), so size is not checked here; corrupt images
    # are caught by the image validation step instead.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_file, session, row.image_url,
                        images_dir / f"{row.isic_id}.jpg"): row.isic_id
            for row in todo
        }
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                records.append(future.result())
            except DownloadError as exc:
                failures.append({"isic_id": futures[future], "error": str(exc)})
            if i % 500 == 0:
                logger.info("Downloaded %d / %d images", i, len(todo))
    return records, failures


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = params["ingest"][DATASET]
    out_dir = raw_dataset_dir(params, DATASET)
    images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(params["http"], pool_size=cfg["max_workers"])
    metadata = build_metadata(session, cfg)
    metadata.to_csv(out_dir / "metadata.csv", index=False)

    done = {row["file"]: row for row in read_manifest(manifest_path)}
    records, failures = download_images(session, metadata, images_dir, done, cfg["max_workers"])
    write_manifest(records, manifest_path)

    failures_path = out_dir / "failures.csv"
    if failures:
        with failures_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["isic_id", "error"])
            writer.writeheader()
            writer.writerows(failures)
    else:
        failures_path.unlink(missing_ok=True)

    failure_rate = len(failures) / len(metadata)
    summary = {
        "dataset": DATASET,
        "listed": len(metadata),
        "images": len(records),
        "failed": len(failures),
        "per_stratum": metadata["stratum"].value_counts().to_dict(),
    }
    logger.info("ISIC ingest complete: %s", summary)
    if failure_rate > cfg.get("max_failure_rate", 0.01):
        raise DownloadError(
            f"{len(failures)} of {len(metadata)} ISIC images failed "
            f"({failure_rate:.1%}); see {failures_path}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=None, help="Path to params.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap every stratum at N images (quick local test)")
    args = parser.parse_args()
    configure_console_logging()
    params = load_params(args.params)
    if args.limit is not None:
        for stratum in params["ingest"][DATASET]["strata"]:
            stratum["max_images"] = min(stratum["max_images"], args.limit)
    run(params)


if __name__ == "__main__":
    main()
