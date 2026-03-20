"""
KazeFlow Prerequisites Downloader.

Checks for required model weights (SPIN v2, RMVPE, SmartCutter) and downloads
them automatically if missing. Called once at app startup.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger("kazeflow.tools")

# ── Paths ──────────────────────────────────────────────────────────────────────
# All paths are relative to the KazeFlow project root (where app.py lives).
# We resolve them at runtime relative to this file's location (2 levels up).
_ROOT = Path(__file__).parent.parent.parent  # kazeflow/tools/ -> kazeflow/ -> root

RMVPE_PATH = _ROOT / "kazeflow" / "models" / "pretrained" / "predictors" / "rmvpe.pt"
SPIN_V2_DIR = _ROOT / "kazeflow" / "models" / "pretrained" / "embedders" / "spin_v2"
SMARTCUTTER_DIR = _ROOT / "kazeflow" / "models" / "pretrained" / "smartcutter"

# ── Download manifests ─────────────────────────────────────────────────────────
# Each entry: (local_path, url)
_PREREQUISITES: list[tuple[Path, str]] = [
    (
        RMVPE_PATH,
        "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/predictors/rmvpe.pt",
    ),
    (
        SPIN_V2_DIR / "pytorch_model.bin",
        "https://huggingface.co/dr87/spinv2_rvc/resolve/main/pytorch_model.bin",
    ),
    (
        SPIN_V2_DIR / "config.json",
        "https://huggingface.co/dr87/spinv2_rvc/resolve/main/config.json",
    ),
    (
        SMARTCUTTER_DIR / "v3_model_32000.pth",
        "https://huggingface.co/Codename0/SmartCutter/resolve/main/v3_model_32000.pth",
    ),
    (
        SMARTCUTTER_DIR / "v3_model_40000.pth",
        "https://huggingface.co/Codename0/SmartCutter/resolve/main/v3_model_40000.pth",
    ),
    (
        SMARTCUTTER_DIR / "v3_model_48000.pth",
        "https://huggingface.co/Codename0/SmartCutter/resolve/main/v3_model_48000.pth",
    ),
]


# ── Download helpers ───────────────────────────────────────────────────────────

def _get_remote_size(url: str) -> int:
    """Return content-length of a remote file (0 if unknown)."""
    try:
        r = requests.head(url, timeout=10, allow_redirects=True)
        return int(r.headers.get("content-length", 0))
    except Exception:
        return 0


def _download_file(url: str, dest: Path, global_bar: tqdm) -> None:
    """Stream-download *url* to *dest*, updating *global_bar* as bytes arrive."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                global_bar.update(len(chunk))
        tmp.rename(dest)
        logger.info("Downloaded %s", dest.name)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        logger.error("Failed to download %s: %s", url, exc)
        raise


# ── Public API ─────────────────────────────────────────────────────────────────

def check_and_download_prerequisites() -> None:
    """
    Check for required model weights and download any that are missing.

    Downloads in parallel with a single combined progress bar.
    Skips files that already exist on disk.
    """
    missing = [(path, url) for path, url in _PREREQUISITES if not path.exists()]

    if not missing:
        logger.info("All prerequisites already present — skipping download.")
        return

    logger.info(
        "%d prerequisite file(s) missing — calculating download size…", len(missing)
    )

    total_size = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        sizes = list(pool.map(lambda x: _get_remote_size(x[1]), missing))
    total_size = sum(sizes)

    desc = "Downloading prerequisites"
    with tqdm(total=total_size or None, unit="iB", unit_scale=True, desc=desc) as bar:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(_download_file, url, path, bar)
                for path, url in missing
            ]
            for f in futures:
                f.result()  # re-raise any download errors

    logger.info("All prerequisites downloaded successfully.")
