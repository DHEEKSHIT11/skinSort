"""Download PAD-UFES-20 from Mendeley Data (doi:10.17632/zr7vgbcyr2).

PAD-UFES-20 is the only dataset with smartphone photos and patient answers
(itch, grew, hurt, changed, bleed, ...), so it is the closest match to SkinSort users.

Output (data/raw/pad_ufes_20/):
    metadata.csv   one row per image, including the patient answers
    images/        PAT_<patient>_<lesion>_<n>.png (~2,300 images)
    manifest.csv   checksum of every downloaded file
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from scripts.common.config import load_params, raw_dataset_dir
from scripts.common.files import extract_images, read_manifest, write_manifest
from scripts.common.http_utils import DownloadError, build_session, download_file, get_json
from scripts.common.logging_utils import configure_console_logging, get_logger

DATASET = "pad_ufes_20"
logger = get_logger(f"ingest.{DATASET}")


def list_mendeley_files(session, api_url: str, dataset_id: str, version: int) -> List[Dict]:
    """List every file in a public Mendeley dataset (root folder plus sub-folders)."""
    base = f"{api_url}/{dataset_id}"
    folders = get_json(session, f"{base}/folders/{version}")
    folder_ids = ["root"] + [folder["id"] for folder in folders]
    files = []
    for folder_id in folder_ids:
        files.extend(
            get_json(session, f"{base}/files", params={"folder_id": folder_id, "version": version})
        )
    if not files:
        raise DownloadError(f"No files listed for Mendeley dataset {dataset_id} v{version}")
    return files


def run(params: Dict[str, Any]) -> Dict[str, Any]:
    cfg = params["ingest"][DATASET]
    out_dir = raw_dataset_dir(params, DATASET)
    images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(params["http"])
    remote = list_mendeley_files(
        session, cfg["mendeley_api_url"], cfg["dataset_id"], cfg["version"]
    )
    done = {row["sha256"]: row for row in read_manifest(manifest_path)}
    records = []

    for entry in remote:
        name = entry["filename"]
        details = entry["content_details"]
        expected_sha = details["sha256_hash"]
        dest = out_dir / name
        marker = out_dir / f".{name}.done"

        if expected_sha in done and (dest.exists() or marker.exists()):
            logger.info("Skipping %s (already downloaded)", name)
            records.append(done[expected_sha])
            continue

        logger.info("Downloading %s (%s bytes)", name, entry.get("size"))
        record = download_file(
            session, details["download_url"], dest, expected_sha, "sha256", entry.get("size")
        )
        records.append(record)

        if dest.suffix == ".zip":
            images = extract_images(dest, images_dir)
            logger.info("Extracted %d images from %s", len(images), name)
            marker.touch()
            if not cfg.get("keep_archives", False):
                dest.unlink()

    write_manifest(records, manifest_path)
    if not (out_dir / "metadata.csv").is_file():
        raise DownloadError("PAD-UFES-20 metadata.csv is missing after download")
    n_images = sum(1 for _ in images_dir.glob("*.png"))
    if n_images == 0:
        raise DownloadError("PAD-UFES-20 download finished but no images were found")
    summary = {"dataset": DATASET, "files": len(records), "images": n_images}
    logger.info("PAD-UFES-20 ingest complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=None, help="Path to params.yaml")
    args = parser.parse_args()
    configure_console_logging()
    run(load_params(args.params))


if __name__ == "__main__":
    main()
