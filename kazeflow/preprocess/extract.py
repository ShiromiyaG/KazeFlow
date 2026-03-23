"""
KazeFlow Feature Extraction.

Ported from Codename-fork-4/rvc/train/extract/extract.py.

Key differences from the original:
- RMVPE only (crepe and fcpe removed — KazeFlow uses RMVPE exclusively).
- No sys.path manipulation / no __main__ CLI entrypoint.
- Uses kazeflow.models.predictors.rmvpe.RMVPE0Predictor directly.
- Uses kazeflow.preprocess.audio.load_audio (librosa-based 16 kHz loader).
- load_embedding is replaced by a local helper that mirrors the original but
  references kazeflow's local SPIN v2 pretrained dir.
- All sys.exit() → RuntimeError/ValueError raises.
"""

import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tqdm

logger = logging.getLogger("kazeflow.preprocess.extract")

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent  # kazeflow/ -> project root
_RMVPE_PATH = (
    _ROOT / "kazeflow" / "models" / "pretrained" / "predictors" / "rmvpe.pt"
)
_SPIN_V2_DIR = (
    _ROOT / "kazeflow" / "models" / "pretrained" / "embedders" / "spin_v2"
)


# ── Audio loading helper ───────────────────────────────────────────────────────

def _load_audio_16k(file_path: str) -> np.ndarray:
    """Load audio file as mono float32 numpy array resampled to 16 kHz."""
    from kazeflow.preprocess.audio import load_audio
    return load_audio(file_path, sample_rate=16000)


# ── Embedding loader ───────────────────────────────────────────────────────────

def _load_embedding(device: str, embedder_name: str = "spin_v2") -> torch.nn.Module:
    """
    Load a content embedder (SPIN v2 or RSPIN).

    Returns a model with forward(wav) → (B, T_frames, output_dim).
    """
    from kazeflow.models.embedder import load_content_embedder
    return load_content_embedder(name=embedder_name, device=device)


# ── F0 extraction ──────────────────────────────────────────────────────────────

class FeatureInput:
    """
    Extracts F0 (pitch) features from 16 kHz audio using RMVPE.

    Args:
        device: PyTorch device string (e.g. "cuda:0", "cpu").
    """

    F0_MIN = 50.0
    F0_MAX = 1100.0
    F0_BIN = 256
    SAMPLE_RATE = 16000
    HOP_SIZE = 160

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.f0_mel_min = 1127 * np.log(1 + self.F0_MIN / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.F0_MAX / 700)
        self._rmvpe: Optional[object] = None

    def _get_rmvpe(self):
        if self._rmvpe is None:
            from kazeflow.models.predictors.rmvpe import RMVPE0Predictor
            if not _RMVPE_PATH.exists():
                raise FileNotFoundError(
                    f"RMVPE weights not found at {_RMVPE_PATH}. "
                    "Run app.py to trigger the automatic prerequisites download."
                )
            self._rmvpe = RMVPE0Predictor(str(_RMVPE_PATH), device=self.device)
        return self._rmvpe

    def compute_f0(self, audio_np: np.ndarray) -> np.ndarray:
        """Return raw F0 (float64, Hz, unvoiced=0) from 16 kHz audio."""
        rmvpe = self._get_rmvpe()
        f0 = rmvpe.infer_from_audio(audio_np, thred=0.03)
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0  # float64, Hz

    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """Convert Hz F0 to coarse mel-scale integer bins (1..F0_BIN-1, 0=unvoiced)."""
        f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
        f0_mel = np.clip(
            (f0_mel - self.f0_mel_min)
            * (self.F0_BIN - 2)
            / (self.f0_mel_max - self.f0_mel_min)
            + 1,
            1,
            self.F0_BIN - 1,
        )
        return np.rint(f0_mel).astype(int)

    def process_file(self, file_info: list) -> None:
        """
        Extract F0 for one file.

        Args:
            file_info: [wav_path, f0_coarse_out_path, f0_full_out_path, _embedding_out]
        """
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return
        try:
            audio_np = _load_audio_16k(inp_path)
            f0_full = self.compute_f0(audio_np)
            np.save(opt_path_full, f0_full, allow_pickle=False)
            f0_coarse = self.coarse_f0(f0_full)
            np.save(opt_path_coarse, f0_coarse, allow_pickle=False)
        except Exception as exc:
            logger.error(
                "Error extracting F0 for %s on %s: %s", inp_path, self.device, exc
            )


def _process_files_worker(files: list, device: str, _threads: int) -> None:
    """Worker function run in a subprocess: extracts F0 for a shard of files."""
    fe = FeatureInput(device=device)
    with tqdm.tqdm(total=len(files), leave=True, desc=f"F0 [{device}]") as pbar:
        for file_info in files:
            fe.process_file(file_info)
            pbar.update(1)


def run_pitch_extraction(
    files: list,
    devices: list[str],
    threads: int,
) -> None:
    """
    Extract F0 for all files, sharding across devices.

    Args:
        files:   List of [wav_path, f0_coarse_out, f0_full_out, embed_out].
        devices: List of device strings (e.g. ["cuda:0", "cpu"]).
        threads: Total thread budget (split evenly across devices).
    """
    devices_str = ", ".join(devices)
    logger.info(
        "Starting F0 extraction with %d threads on %s (RMVPE)...",
        threads, devices_str,
    )
    start = time.time()
    n = len(devices)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
        futures = [
            executor.submit(
                _process_files_worker,
                files[i::n],
                devices[i],
                max(1, threads // n),
            )
            for i in range(n)
        ]
        concurrent.futures.wait(futures)
        # Re-raise any subprocess exceptions
        for fut in futures:
            fut.result()
    logger.info("F0 extraction completed in %.2fs.", time.time() - start)


# ── Embedding extraction ───────────────────────────────────────────────────────

def _process_file_embedding_worker(
    files: list,
    device_num: int,
    device: str,
    n_threads: int,
    embedder_name: str = "spin_v2",
) -> None:
    """Worker: extract content embeddings for a shard of files."""
    model = _load_embedding(device, embedder_name)
    expected_dim = model.output_dim
    n_threads = max(1, n_threads)

    def _worker(file_info):
        wav_path, _, _, out_path = file_info
        if os.path.exists(out_path):
            # Check dimension matches current embedder; re-extract on mismatch
            try:
                old = np.load(out_path)
                if old.shape[-1] == expected_dim:
                    return
                logger.info("Dim mismatch %s: got %d, need %d — re-extracting",
                            out_path, old.shape[-1], expected_dim)
            except Exception:
                pass  # corrupted file — re-extract
        try:
            audio_np = _load_audio_16k(wav_path)
            feats = torch.from_numpy(audio_np).to(device).float().view(1, -1)
            with torch.no_grad():
                result = model(feats)  # (1, T, embed_dim)
            feats_out = result.squeeze(0).float().cpu().numpy()  # (T, embed_dim)
            if not np.isnan(feats_out).any():
                np.save(out_path, feats_out, allow_pickle=False)
            else:
                logger.warning("%s produced NaN values; skipping.", wav_path)
        except Exception as exc:
            logger.error("Embedding error for %s: %s", wav_path, exc)

    with tqdm.tqdm(
        total=len(files), leave=True, position=device_num,
        desc=f"Embed [{device}]"
    ) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(_worker, f) for f in files]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def run_embedding_extraction(
    files: list,
    devices: list[str],
    threads: int,
    embedder_name: str = "spin_v2",
) -> None:
    """
    Extract content embeddings for all files, sharding across devices.

    Args:
        files:          List of [wav_path, f0_coarse_out, f0_full_out, embed_out].
        devices:        List of device strings.
        threads:        Total thread budget (split evenly across devices).
        embedder_name:  "spin_v2" or "rspin".
    """
    devices_str = ", ".join(devices)
    logger.info(
        "Starting embedding extraction with %d threads on %s (%s)...",
        threads, devices_str, embedder_name,
    )
    start = time.time()
    n = len(devices)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n) as executor:
        futures = [
            executor.submit(
                _process_file_embedding_worker,
                files[i::n],
                i,
                devices[i],
                max(1, threads // n),
                embedder_name,
            )
            for i in range(n)
        ]
        concurrent.futures.wait(futures)
        for fut in futures:
            fut.result()
    logger.info("Embedding extraction completed in %.2fs.", time.time() - start)
