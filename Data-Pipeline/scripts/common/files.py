"""File helpers: checksums, manifests and safe archive extraction."""

import csv
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MANIFEST_FIELDS = ["file", "bytes", "sha256", "source_url"]


class ArchiveError(Exception):
    """Raised when a zip archive is corrupt or unsafe to extract."""


def file_hash(path: Path, algorithm: str = "sha256", chunk_size: int = 1 << 20) -> str:
    """Return the hex digest of a file, read in chunks so large files fit in memory."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(records: Iterable[Dict], path: Path) -> int:
    """Write download records (file, bytes, sha256, source_url) to a CSV manifest."""
    rows = sorted(records, key=lambda r: r["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_manifest(path: Path) -> List[Dict]:
    """Read a manifest written by write_manifest; returns [] if it does not exist."""
    if not path.is_file():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def extract_images(archive: Path, dest_dir: Path) -> List[Path]:
    """Extract only image files from a zip into dest_dir, flattening sub-folders.

    Flattening keeps one predictable layout (images/<file>) across datasets.
    Members are written by basename only, so a malicious path such as
    '../../etc/x' cannot escape dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    try:
        with zipfile.ZipFile(archive) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                raise ArchiveError(f"Corrupt member {bad_member!r} in {archive}")
            for member in zf.infolist():
                name = Path(member.filename).name
                if member.is_dir() or name.startswith("."):
                    continue
                if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                target = dest_dir / name
                with zf.open(member) as src, target.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                extracted.append(target)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"Not a valid zip archive: {archive}") from exc
    return extracted
