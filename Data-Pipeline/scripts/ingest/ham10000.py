"""Download HAM10000 from Harvard Dataverse (doi:10.7910/DVN/DBW86T).

Output (data/raw/ham10000/):
    metadata.csv   lesion_id, image_id, dx, dx_type, age, sex, localization, dataset
    images/        ISIC_XXXXXXX.jpg (10,015 dermoscopic images)
    manifest.csv   checksum of every downloaded file
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from scripts.common.config import load_params, raw_dataset_dir
from scripts.common.files import extract_images, read_manifest, write_manifest
from scripts.common.http_utils import DownloadError, build_session, download_file, get_json
from scripts.common.logging_utils import configure_console_logging, get_logger

DATASET = "ham10000"
logger = get_logger(f"ingest.{DATASET}")


def list_dataverse_files(session, base_url: str, doi: str) -> Dict[str, Dict[str, Any]]:
    """Return {filename: dataFile} for the latest version of a Dataverse dataset."""
    url = f"{base_url}/api/datasets/:persistentId/"
    payload = get_json(session, url, params={"persistentId": doi})
    try:
        files = payload["data"]["latestVersion"]["files"]
    except (KeyError, TypeError) as exc:
        raise DownloadError(f"Unexpected Dataverse response for {doi}") from exc
    return {f["dataFile"]["filename"]: f["dataFile"] for f in files}


def _local_name(filename: str) -> str:
    # Dataverse stores the metadata CSV as a tabular ".tab" file; we ask for the
    # original CSV and save it under a clearer name.
    return "metadata.csv" if filename.endswith("metadata.tab") else filename


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = params["ingest"][DATASET]
    out_dir = raw_dataset_dir(params, DATASET)
    images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(params["http"])
    remote = list_dataverse_files(session, cfg["dataverse_url"], cfg["doi"])
    missing = [name for name in cfg["files"] if name not in remote]
    if missing:
        raise DownloadError(f"Files not found on Dataverse: {missing}")

    done = {row["source_url"]: row for row in read_manifest(manifest_path)}
    records: List[Dict[str, Any]] = []

    for name in cfg["files"]:
        data_file = remote[name]
        url = f"{cfg['dataverse_url']}/api/access/datafile/{data_file['id']}"
        if name.endswith(".tab"):
            url += "?format=original"
        dest = out_dir / _local_name(name)
        marker = out_dir / f".{dest.name}.done"

        if url in done and (dest.exists() or marker.exists()):
            logger.info("Skipping %s (already downloaded)", name)
            records.append(done[url])
            continue

        checksum = data_file.get("checksum", {})
        # Tabular files are re-generated on request, so their stored checksum
        # refers to the ingested .tab, not the original CSV we download.
        expected = None if name.endswith(".tab") else checksum.get("value")
        algorithm = checksum.get("type", "MD5").lower().replace("-", "")
        logger.info("Downloading %s (%s bytes)", name, data_file.get("filesize"))
        record = download_file(session, url, dest, expected, algorithm)
        records.append(record)

        if dest.suffix == ".zip":
            images = extract_images(dest, images_dir)
            logger.info("Extracted %d images from %s", len(images), name)
            marker.touch()
            if not cfg.get("keep_archives", False):
                dest.unlink()

    write_manifest(records, manifest_path)
    n_images = sum(1 for _ in images_dir.glob("*.jpg"))
    if n_images == 0:
        raise DownloadError("HAM10000 download finished but no images were found")
    summary = {"dataset": DATASET, "files": len(records), "images": n_images}
    logger.info("HAM10000 ingest complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=None, help="Path to params.yaml")
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
