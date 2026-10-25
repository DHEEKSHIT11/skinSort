"""HTTP session with retries, and a checksum-verified file downloader."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.common.logging_utils import get_logger

logger = get_logger("http_utils")

RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


class DownloadError(Exception):
    """Raised when a file cannot be downloaded or fails verification."""


def build_session(http_params: Dict[str, Any], pool_size: int = 16) -> requests.Session:
    """Create a requests session that retries on network errors and 429/5xx responses."""
    retry = Retry(
        total=http_params.get("max_retries", 5),
        backoff_factor=http_params.get("backoff_factor", 1.0),
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_size, pool_maxsize=pool_size)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = http_params.get("user_agent", "SkinSort-DataPipeline")
    session.request_timeout = http_params.get("timeout_seconds", 60)
    return session


def get_json(session: requests.Session, url: str, **kwargs) -> Any:
    """GET a URL and decode JSON, raising DownloadError with context on failure."""
    try:
        response = session.get(url, timeout=session.request_timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DownloadError(f"Failed to fetch JSON from {url}: {exc}") from exc


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    expected_hash: Optional[str] = None,
    hash_algorithm: str = "sha256",
    expected_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Stream a URL to ``dest`` and verify its size and checksum.

    The file is written to ``dest.part`` first and only renamed when complete
    and verified, so an interrupted run never leaves a half-written file that
    looks finished. Returns a manifest record for the file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.new(hash_algorithm)
    sha256 = hashlib.sha256() if hash_algorithm != "sha256" else digest
    size = 0
    try:
        with session.get(url, stream=True, timeout=session.request_timeout) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    digest.update(chunk)
                    if sha256 is not digest:
                        sha256.update(chunk)
                    size += len(chunk)
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Download failed for {url}: {exc}") from exc

    if size == 0:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Empty response from {url}")
    if expected_size is not None and size != int(expected_size):
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Size mismatch for {url}: expected {expected_size}, got {size}")
    if expected_hash and digest.hexdigest().lower() != expected_hash.lower():
        tmp.unlink(missing_ok=True)
        raise DownloadError(
            f"{hash_algorithm} mismatch for {url}: expected {expected_hash}, "
            f"got {digest.hexdigest()}"
        )

    tmp.replace(dest)
    return {"file": dest.name, "bytes": size, "sha256": sha256.hexdigest(), "source_url": url}
