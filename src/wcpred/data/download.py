"""Download the raw results dataset at a pinned upstream commit."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

from wcpred import config as C
from wcpred.data.results import DataError

log = logging.getLogger(__name__)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_results(ref: str = C.RESULTS_REF, dest_dir: Path | None = None) -> list[Path]:
    """Download results.csv and shootouts.csv from martj42/international_results.

    At the pinned default ref the files are checked against known SHA-256
    hashes, so a changed upstream file cannot silently change results.
    """
    dest_dir = dest_dir or C.PATHS.raw_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, expected in C.RESULTS_SHA256.items():
        url = f"{C.RESULTS_REPO_RAW}/{ref}/{name}"
        log.info("downloading %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "wcpred"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        except OSError as exc:
            raise DataError(
                f"could not download {url}: {exc}. If you are offline, place the CSV "
                f"in {dest_dir} manually (schema in data/README.md)."
            ) from exc
        digest = sha256(data)
        if ref == C.RESULTS_REF and expected and digest != expected:
            raise DataError(f"checksum mismatch for {name}: got {digest}, expected {expected}")
        path = dest_dir / name
        path.write_bytes(data)
        log.info("wrote %s (%s bytes, sha256 %s)", path, f"{len(data):,}", digest[:12])
        written.append(path)
    return written
