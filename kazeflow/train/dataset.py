"""
Dataset and DataLoader for KazeFlow training.

Expected file structure after preprocessing:
    dataset_root/
        filelist.txt          # lines: "wav_path|spk_id"
        sliced_audios/        # GT waveforms (.wav, target SR)
        spin/                 # SPIN v2 features (.npy, shape [T, 768])
        f0/                   # F0 contours (.npy, shape [T])
        mel/                  # Mel spectrograms (.npy, shape [n_mels, T])

Mute entries (stem == "mute") are loaded from the shared mute directory
``logs/mute_spin_v2/`` or ``logs/mute_rspin/`` depending on the content embedder.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("kazeflow.dataset")

# Shared mute feature directory (generated once at startup by generate_mutes.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_MUTE_DIRS = {
    "spin_v2": _PROJECT_ROOT / "logs" / "mute_spin_v2",
    "rspin":   _PROJECT_ROOT / "logs" / "mute_rspin",
}


class KazeFlowDataset(Dataset):
    """
    Loads preprocessed features + GT audio for KazeFlow training.

    Each sample: (mel, spin_features, f0, speaker_id, wav_segment)

    The GT waveform segment is loaded from ``sliced_audios/`` and cropped
    to the same temporal region as the mel/spin/f0 features.  This allows
    the discriminator to compare vocoder output against real audio directly,
    avoiding a second vocoder forward pass (which would double VRAM usage).

    Random segment cropping is done here for efficiency.
    """

    def __init__(
        self,
        filelist_path: str,
        dataset_root: str,
        segment_frames: int = 48,
        n_mels: int = 128,
        hop_length: int = 480,
        sample_rate: int = 48000,
        skip_wav: bool = False,
        content_embedder: str = "spin_v2",
    ):
        self.dataset_root = Path(dataset_root)
        self.segment_frames = segment_frames
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.skip_wav = skip_wav
        self.mute_dir = _MUTE_DIRS.get(content_embedder, _MUTE_DIRS["spin_v2"])

        # Parse filelist
        self.entries = []
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    wav_path = parts[0].strip()
                    spk_id = int(parts[1].strip())
                    self.entries.append((wav_path, spk_id))

    def __len__(self):
        return len(self.entries)

    def _load_wav(self, feature_root: Path, stem: str) -> torch.Tensor:
        """Load GT waveform from sliced_audios/. Returns (T_samples,) float32."""
        # Mute files are named mute{sr}.wav
        if stem == "mute":
            wav_path = feature_root / "sliced_audios" / f"mute{self.sample_rate}.wav"
        else:
            wav_path = feature_root / "sliced_audios" / f"{stem}.wav"

        if wav_path.exists():
            audio, sr = sf.read(str(wav_path), dtype="float32")
            return torch.from_numpy(audio).float()
        else:
            # Fallback: return zeros (shouldn't happen in normal use)
            logger.warning("GT audio not found: %s — using silence", wav_path)
            n_samples = self.segment_frames * self.hop_length
            return torch.zeros(n_samples, dtype=torch.float32)

    def __getitem__(self, idx):
        wav_path, spk_id = self.entries[idx]
        # wav_path can be a full path or just a stem name
        stem = Path(wav_path).stem

        # Mute entries are stored in the shared mute directory, not the model dir.
        if stem == "mute":
            feature_root = self.mute_dir
        else:
            feature_root = self.dataset_root

        # Load preprocessed features
        mel = np.load(feature_root / "mel" / f"{stem}.npy")        # (n_mels, T)
        spin = np.load(feature_root / "spin" / f"{stem}.npy")      # (T, 768)
        f0 = np.load(feature_root / "f0" / f"{stem}.npy")          # (T,)

        mel = torch.from_numpy(mel).float()
        spin = torch.from_numpy(spin).float().T    # (768, T_spin)
        f0 = torch.from_numpy(f0).float()

        # Load GT waveform (skip during warmup to reduce SSD I/O)
        if self.skip_wav:
            wav = torch.zeros(self.segment_frames * self.hop_length,
                              dtype=torch.float32)
        else:
            wav = self._load_wav(feature_root, stem)  # (T_samples,)

        # SPIN v2 runs at 50Hz (20ms hop @ 16kHz) while mel/F0 run at 100Hz
        # (10ms hop).  Interpolate SPIN to mel frame rate so we don't discard
        # half of the mel/F0 frames via min_len truncation.
        target_len = mel.shape[1]
        if spin.shape[1] < target_len:
            spin = F.interpolate(
                spin.unsqueeze(0), size=target_len, mode="linear",
                align_corners=False,
            ).squeeze(0)

        # Align feature lengths to minimum
        min_len = min(mel.shape[1], spin.shape[1], f0.shape[0])
        mel = mel[:, :min_len]
        spin = spin[:, :min_len]
        f0 = f0[:min_len]

        # Random segment crop
        if min_len > self.segment_frames:
            start = random.randint(0, min_len - self.segment_frames)
            mel = mel[:, start:start + self.segment_frames]
            spin = spin[:, start:start + self.segment_frames]
            f0 = f0[start:start + self.segment_frames]

            # Crop GT audio to match (frame offset → sample offset)
            audio_start = start * self.hop_length
            audio_len = self.segment_frames * self.hop_length
            wav = wav[audio_start:audio_start + audio_len]
        else:
            # Pad if too short
            pad_len = self.segment_frames - min_len
            mel = F.pad(mel, (0, pad_len))
            spin = F.pad(spin, (0, pad_len))
            f0 = F.pad(f0, (0, pad_len))

            # Pad audio to match
            audio_len = self.segment_frames * self.hop_length
            if wav.shape[0] < audio_len:
                wav = F.pad(wav, (0, audio_len - wav.shape[0]))

        # Ensure exact audio length (iSTFT may need exact alignment)
        target_audio_len = self.segment_frames * self.hop_length
        if wav.shape[0] > target_audio_len:
            wav = wav[:target_audio_len]
        elif wav.shape[0] < target_audio_len:
            wav = F.pad(wav, (0, target_audio_len - wav.shape[0]))

        return mel, spin, f0, spk_id, wav


class KazeFlowCollator:
    """Simple collator — all samples are same length after crop/pad."""

    def __call__(self, batch):
        mels, spins, f0s, spk_ids, wavs = zip(*batch)
        mels = torch.stack(mels, dim=0)      # (B, n_mels, T)
        spins = torch.stack(spins, dim=0)     # (B, 768, T)
        f0s = torch.stack(f0s, dim=0)         # (B, T)
        spk_ids = torch.tensor(spk_ids, dtype=torch.long)
        wavs = torch.stack(wavs, dim=0).unsqueeze(1)  # (B, 1, T_audio)
        return mels, spins, f0s, spk_ids, wavs


def create_dataloader(
    filelist_path: str,
    dataset_root: str,
    batch_size: int = 8,
    segment_frames: int = 48,
    n_mels: int = 128,
    hop_length: int = 480,
    sample_rate: int = 48000,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    skip_wav: bool = False,
    content_embedder: str = "spin_v2",
) -> DataLoader:
    dataset = KazeFlowDataset(
        filelist_path=filelist_path,
        dataset_root=dataset_root,
        segment_frames=segment_frames,
        n_mels=n_mels,
        hop_length=hop_length,
        sample_rate=sample_rate,
        skip_wav=skip_wav,
        content_embedder=content_embedder,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=KazeFlowCollator(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
