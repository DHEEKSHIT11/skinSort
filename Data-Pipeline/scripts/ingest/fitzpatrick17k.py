"""Download Fitzpatrick17k (github.com/mattgroh/fitzpatrick17k).

Used only to test fairness across skin tones; never used for training.
The images are hosted on third-party atlas websites, so they are kept in our
private bucket only and are never redistributed. Some links are dead, which
is expected: failures are logged and the task only fails if too many are lost.

Each CSV row carries the MD5 hash of its image, which we use to verify downloads.

Output (data/raw/fitzpatrick17k/):
    metadata.csv   md5hash, fitzpatrick_scale, label, partitions, url, ...
    images/        <md5hash>.jpg
    manifest.csv   checksum of every downloaded image
    failures.csv   rows whose image could not be downloaded
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from scripts.common.config import load_params, raw_dataset_dir
from scripts.common.files import read_manifest, write_manifest
from scripts.common.http_utils import DownloadError, build_session, download_file
from scripts.common.logging_utils import configure_console_logging, get_logger

DATASET = "fitzpatrick17k"
logger = get_logger(f"ingest.{DATASET}")

REQUIRED_COLUMNS = {"md5hash", "fitzpatrick_scale", "label", "url"}


def load_metadata(session, csv_url: str, dest: Path) -> pd.DataFrame:
    """Download the label CSV and check it has the columns we rely on."""
    download_file(session, csv_url, dest)
    metadata = pd.read_csv(dest)
    missing = REQUIRED_COLUMNS - set(metadata.columns)
    if missing:
        raise DownloadError(f"Fitzpatrick17k CSV is missing columns: {sorted(missing)}")
    return metadata


def load_permanent_failures(path: Path) -> Dict[str, Dict[str, str]]:
    """URLs that returned HTTP 404 on a previous run, keyed by URL.

    A 404 means the image was removed, so retrying it only wastes time (each
    dead dermaamin.com link takes ~8 s to answer). Timeouts and 5xx errors are
    not cached: those may succeed next time.
    """
    if not path.is_file():
        return {}
    previous = pd.read_csv(path, dtype=str).fillna("")
    # Match the HTTP error text ("404 Client Error"), not a 404 inside the URL.
    gone = previous[previous["error"].str.contains(r"\b404 (?:Client )?Error", regex=True)]
    return {row["url"]: row.to_dict() for _, row in gone.iterrows()}


def run(params: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, Any]:
    cfg = params["ingest"][DATASET]
    out_dir = raw_dataset_dir(params, DATASET)
    images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(params["http"], pool_size=cfg["max_workers"])
    metadata = load_metadata(session, cfg["csv_url"], out_dir / "metadata.csv")
    rows = metadata.dropna(subset=["url", "md5hash"]).drop_duplicates(subset="md5hash")
    if limit is not None:
        rows = rows.head(limit)

    failures_path = out_dir / "failures.csv"
    done = {row["file"]: row for row in read_manifest(manifest_path)}
    known_gone = load_permanent_failures(failures_path) if cfg.get("skip_known_404", True) else {}
    records, failures, todo = [], [], []
    for row in rows.itertuples(index=False):
        name = f"{row.md5hash}.jpg"
        if name in done and (images_dir / name).exists():
            records.append(done[name])
        elif row.url in known_gone:
            failures.append(known_gone[row.url])
        else:
            todo.append(row)
    logger.info("%d images already present, %d known dead links skipped, %d to download",
                len(records), len(failures), len(todo))

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as pool:
        futures = {
            pool.submit(download_file, session, row.url, images_dir / f"{row.md5hash}.jpg",
                        row.md5hash, "md5"): row
            for row in todo
        }
        for i, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                records.append(future.result())
            except DownloadError as exc:
                failures.append({"md5hash": row.md5hash, "url": row.url, "error": str(exc)})
            if i % 1000 == 0:
                logger.info("Processed %d / %d images", i, len(todo))

    write_manifest(records, manifest_path)
    with failures_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["md5hash", "url", "error"])
        writer.writeheader()
        writer.writerows(failures)

    success_rate = len(records) / len(rows) if len(rows) else 0.0
    summary = {
        "dataset": DATASET,
        "listed": len(rows),
        "images": len(records),
        "failed": len(failures),
        "success_rate": round(success_rate, 4),
    }
    logger.info("Fitzpatrick17k ingest complete: %s", summary)
    if success_rate < cfg["min_success_rate"]:
        raise DownloadError(
            f"Only {success_rate:.1%} of Fitzpatrick17k images downloaded "
            f"(minimum {cfg['min_success_rate']:.0%}); see {failures_path}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=None, help="Path to params.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only download the first N images (quick local test)")
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params), limit=args.limit)


if __name__ == "__main__":
    main()
