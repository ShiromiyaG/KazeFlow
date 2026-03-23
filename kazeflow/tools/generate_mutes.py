"""
KazeFlow Mute File Generator.

Generates the mute reference files used by generate_filelist() to pad the
training filelist with silence samples.  These improve training stability by
giving the model explicit examples of silence mapped to a known output.

Output layout::

    logs/mute_spin_v2/
        sliced_audios/
            mute32000.wav  …
        spin/
            mute.npy          # (T_spin, 768) float32 — SPIN v2 on 3s silence @ 48k
        f0/
            mute.npy          # (T_f0,)  float64 — all zeros (unvoiced)
        mel/
            mute.npy          # (n_mels, T_mel) float32 — log-mel of silence

    logs/mute_rspin/
        sliced_audios/ …     # same wavs
        spin/
            mute.npy          # (T_spin, 256) float32 — RSPIN on 3s silence @ 48k
        f0/
            mute.npy          # same as spin_v2
        mel/
            mute.npy          # same as spin_v2

The spin / f0 / mel arrays are derived from a 3-second silence clip at 48 kHz
(the most common target SR).  The same mute.npy files are reused regardless of
the actual training SR — the frames count difference is small and intentionally
handled by the dataset's segment-crop logic.

Usage::

    python kazeflow/tools/generate_mutes.py

Run once after cloning or after deleting the logs/ directory.
Requires the SPIN v2 and RMVPE weights to already be present.
"""

import json
import logging
import os
import gc
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("kazeflow.tools.generate_mutes")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent  # kazeflow/tools/ -> kazeflow/ -> project root (wrong; see below)
# tools/ is inside kazeflow/; project root is two levels up
_ROOT = _HERE.parent.parent  # project root
_MUTE_DIRS = {
    "spin_v2": _ROOT / "logs" / "mute_spin_v2",
    "rspin":   _ROOT / "logs" / "mute_rspin",
}

_SPIN_V2_DIR = _ROOT / "kazeflow" / "models" / "pretrained" / "embedders" / "spin_v2"
_RMVPE_PATH = (
    _ROOT / "kazeflow" / "models" / "pretrained" / "predictors" / "rmvpe.pt"
)
_CONFIGS_DIR = _ROOT / "kazeflow" / "configs"

# Duration of the reference silence clip (seconds)
_MUTE_DURATION_S = 3.0
# Primary SR used for spin/f0/mel mute arrays
_PRIMARY_SR = 48000
# All SRs for which a sliced_audios wav is generated
_ALL_SRS = [32000, 40000, 44100, 48000]


def _make_silence(sr: int, duration: float) -> np.ndarray:
    """Return a zero float32 array of *duration* seconds at *sr* Hz."""
    return np.zeros(int(sr * duration), dtype=np.float32)


def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")
    logger.info("Saved %s", path)


def _generate_spin(silence_48k: np.ndarray, device: str, embedder_name: str = "spin_v2") -> np.ndarray:
    """
    Run content embedder on the silence clip and return (T, D) float32.
    D = 768 for spin_v2, 256 for rspin.
    """
    from kazeflow.models.embedder import load_content_embedder

    model = load_content_embedder(name=embedder_name, device=device)

    # Content embedders expect 16 kHz
    audio_16k = torchaudio.functional.resample(
        torch.from_numpy(silence_48k).unsqueeze(0), _PRIMARY_SR, 16000
    ).to(device)

    with torch.no_grad():
        features = model(audio_16k)  # (1, T, 768)
    features = features.squeeze(0).float().cpu().numpy()  # (T, 768)
    logger.info("Mute embedding shape: %s", features.shape)

    # Free model from GPU before returning.
    del model, audio_16k
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return features


def _generate_f0(silence_48k: np.ndarray, device: str) -> np.ndarray:
    """
    Run RMVPE on the silence clip and return (T,) float64 (all zeros = unvoiced).
    """
    from kazeflow.models.predictors.rmvpe import RMVPE0Predictor

    if not _RMVPE_PATH.exists():
        raise FileNotFoundError(
            f"RMVPE weights not found at {_RMVPE_PATH}. "
            "Run app.py to trigger the automatic prerequisites download."
        )
    logger.info("Loading RMVPE…")
    predictor = RMVPE0Predictor(str(_RMVPE_PATH), device=device)

    # RMVPE expects 16 kHz mono float32 numpy
    audio_16k = torchaudio.functional.resample(
        torch.from_numpy(silence_48k).unsqueeze(0), _PRIMARY_SR, 16000
    ).squeeze(0).numpy()

    f0 = predictor.infer_from_audio(audio_16k, thred=0.03)
    f0 = np.nan_to_num(f0, nan=0.0)
    logger.info("F0 mute shape: %s  (should be all zeros)", f0.shape)

    # Free RMVPE from GPU before returning.
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return f0


def _generate_mel(silence_48k: np.ndarray, config: dict, device: str) -> np.ndarray:
    """
    Compute log-mel spectrogram of the silence clip.
    Returns (n_mels, T) float32.
    """
    import torch.nn.functional as F

    model_cfg = config["model"]
    n_fft = model_cfg["n_fft"]
    hop_length = model_cfg["hop_length"]
    win_length = model_cfg["win_length"]
    n_mels = model_cfg["n_mels"]
    sample_rate = model_cfg["sample_rate"]

    audio = torch.from_numpy(silence_48k).float().to(device)

    # Resample if needed (primary SR matches config SR in most cases)
    if sample_rate != _PRIMARY_SR:
        audio = torchaudio.functional.resample(
            audio.unsqueeze(0), _PRIMARY_SR, sample_rate
        ).squeeze(0)

    window = torch.hann_window(win_length, device=device)
    pad = (n_fft - hop_length) // 2
    audio_padded = F.pad(audio.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)

    spec = torch.stft(
        audio_padded.unsqueeze(0),
        n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window,
        center=False, return_complex=True,
    )
    mag = spec.abs().squeeze(0)  # (n_fft//2+1, T)

    mel_basis = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
    ).T.to(device)

    mel = torch.matmul(mel_basis, mag)
    log_mel = torch.log(torch.clamp(mel, min=1e-5)).float().cpu().numpy()
    logger.info("Mel mute shape: %s", log_mel.shape)
    return log_mel


def generate_mutes(device: str = "cpu", force: bool = False) -> None:
    """
    Generate all mute files under logs/mute_spin_v2/ and logs/mute_rspin/.

    Args:
        device: PyTorch device to use for model inference.
        force:  Overwrite existing files even if they already exist.
    """
    # Load 48k config for mel parameters
    from kazeflow.configs import load_config
    config = load_config(48000)

    silence_48k = _make_silence(_PRIMARY_SR, _MUTE_DURATION_S)

    # F0 and mel are shared (embedder-independent) — compute once, reuse.
    f0_arr: np.ndarray | None = None
    mel_arr: np.ndarray | None = None

    for embedder_name, mute_dir in _MUTE_DIRS.items():
        spin_path = mute_dir / "spin" / "mute.npy"
        f0_path = mute_dir / "f0" / "mute.npy"
        mel_path = mute_dir / "mel" / "mute.npy"
        wav_path = mute_dir / "sliced_audios" / f"mute{_PRIMARY_SR}.wav"

        all_exist = all(p.exists() for p in (spin_path, f0_path, mel_path, wav_path))
        if all_exist and not force:
            logger.info("Mute files already present at %s — skipping.", mute_dir)
            continue

        logger.info("Generating mute files for %s…", embedder_name)

        # ── WAV files (all SRs) ───────────────────────────────────────────
        for sr in _ALL_SRS:
            wav = mute_dir / "sliced_audios" / f"mute{sr}.wav"
            if not wav.exists() or force:
                if sr == _PRIMARY_SR:
                    _save_wav(wav, silence_48k, sr)
                else:
                    silence_sr = _make_silence(sr, _MUTE_DURATION_S)
                    _save_wav(wav, silence_sr, sr)

        # ── Content embedder features ─────────────────────────────────────
        if not spin_path.exists() or force:
            spin_features = _generate_spin(silence_48k, device, embedder_name)
            # Zero out embeddings for silence — consistent with energy gate
            # applied during preprocessing (pipeline.py).  Teaches the CFM
            # that null embeddings → silent mel.
            spin_features = np.zeros_like(spin_features)
            spin_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(spin_path, spin_features, allow_pickle=False)
            logger.info("Saved %s %s (zeroed for silence)", spin_path, spin_features.shape)

        # ── F0 (shared — compute once) ────────────────────────────────────
        if not f0_path.exists() or force:
            if f0_arr is None:
                f0_arr = _generate_f0(silence_48k, device)
            f0_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(f0_path, f0_arr, allow_pickle=False)
            logger.info("Saved %s %s", f0_path, f0_arr.shape)

        # ── Mel (shared — compute once) ───────────────────────────────────
        if not mel_path.exists() or force:
            if mel_arr is None:
                mel_arr = _generate_mel(silence_48k, config, device)
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mel_path, mel_arr, allow_pickle=False)
            logger.info("Saved %s %s", mel_path, mel_arr.shape)

    logger.info("Mute file generation complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate KazeFlow mute files.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing mute files.",
    )
    args = parser.parse_args()

    # Add project root to sys.path so kazeflow imports work
    sys.path.insert(0, str(_ROOT))

    generate_mutes(device=args.device, force=args.force)
