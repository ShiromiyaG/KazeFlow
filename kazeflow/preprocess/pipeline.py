"""
KazeFlow Preprocessing Pipeline.

Processes raw audio into training-ready features:
1. Audio loading + resampling to target SR
2. Highpass filtering (48Hz)
3. Loudness normalization
4. Silence trimming / smart cutting
5. SPIN v2 feature extraction (content)
6. F0 extraction
7. Mel spectrogram computation
8. Filelist generation (auto-detects multi-speaker from subdirectories)

Output structure:
    output_dir/
        filelist.txt
        model_info.json
        spin/      *.npy  (T, 768)
        f0/        *.npy  (T,)
        mel/       *.npy  (n_mels, T)

Multi-speaker layout (auto-detected):
    dataset_dir/
        0_SpeakerA/
            audio1.wav
        1_SpeakerB/
            audio2.wav
    OR single-speaker:
    dataset_dir/
        audio1.wav
        audio2.wav
"""

import logging
import os
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import highpass_biquad
from tqdm import tqdm

logger = logging.getLogger("kazeflow.preprocess")


class PreprocessPipeline:
    """
    Preprocesses raw audio files into features for KazeFlow training.
    """

    def __init__(
        self,
        config: dict,
        device: str = "cuda",
    ):
        self.config = config
        self.device = torch.device(device)

        model_cfg = config["model"]
        preprocess_cfg = config["preprocess"]

        self.sample_rate = model_cfg["sample_rate"]
        self.hop_length = model_cfg["hop_length"]
        self.n_fft = model_cfg["n_fft"]
        self.win_length = model_cfg["win_length"]
        self.n_mels = model_cfg["n_mels"]

        self.min_duration = preprocess_cfg.get("min_duration", 0.5)
        self.max_duration = preprocess_cfg.get("max_duration", 15.0)
        self.highpass_freq = preprocess_cfg.get("highpass_freq", 48)
        self.target_db = preprocess_cfg.get("target_db", -23.0)
        self.spin_model_name = preprocess_cfg.get("spin_model", "dr87/spinv2_rvc")
        self.content_embedder_name = preprocess_cfg.get("content_embedder", "spin_v2")

        self._spin_model = None
        self._mel_basis = None
        self._rmvpe_model = None

    def _get_spin_model(self):
        if self._spin_model is None:
            from kazeflow.models.embedder import load_content_embedder
            self._spin_model = load_content_embedder(
                name=self.content_embedder_name,
                device=str(self.device),
                spin_source=self.spin_model_name,
            )
        return self._spin_model

    def _get_mel_basis(self):
        if self._mel_basis is None:
            self._mel_basis = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=0.0,
                f_max=self.sample_rate / 2.0,
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
            ).T.to(self.device)
        return self._mel_basis

    def process_dataset(
        self,
        audio_dir: str,
        output_dir: str,
        speaker_id: int = 0,
        extensions: tuple = (".wav", ".flac", ".mp3", ".ogg"),
    ):
        """
        Process all audio files in a directory.

        Auto-detects multi-speaker layout:
        - If subdirectories named like ``0_Name``, ``1_Name`` exist,
          each is treated as a separate speaker.
        - Otherwise all files use ``speaker_id``.

        Args:
            audio_dir: Directory containing audio files
            output_dir: Output directory for preprocessed features
            speaker_id: Speaker ID for single-speaker mode
            extensions: Audio file extensions to process
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)

        # Create output subdirs
        for subdir in ["spin", "f0", "mel"]:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Detect speakers from directory structure
        audio_entries = self._scan_dataset(audio_dir, speaker_id, extensions)

        if not audio_entries:
            logger.warning(f"No audio files found in {audio_dir}")
            return

        speaker_ids = sorted(set(sid for _, sid in audio_entries))
        logger.info(
            f"Found {len(audio_entries)} audio files, "
            f"{len(speaker_ids)} speaker(s)"
        )

        processed_entries = []

        pbar = tqdm(audio_entries, desc="Preprocessing", unit="file",
                    dynamic_ncols=True)
        for audio_path, spk_id in pbar:
            try:
                stem = self._make_stem(audio_path, spk_id)
                success = self._process_single(
                    audio_path, output_dir, spk_id, stem)
                if success:
                    processed_entries.append((stem, spk_id))
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")

        # Generate filelist via intersection
        self._generate_filelist(output_dir, processed_entries, speaker_ids)

        logger.info(
            f"Preprocessing complete: {len(processed_entries)}/{len(audio_entries)} "
            f"files processed"
        )

        # Free GPU memory used by SPIN v2 and RMVPE — not needed after extraction.
        self.cleanup()

    def cleanup(self):
        """Release SPIN v2, RMVPE, and mel basis from GPU memory."""
        self._spin_model = None
        self._rmvpe_model = None
        self._mel_basis = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("PreprocessPipeline: VRAM released.")

    def _scan_dataset(self, audio_dir: Path, default_spk: int,
                      extensions: tuple):
        """
        Scan dataset directory, auto-detecting multi-speaker layout.

        Multi-speaker mode is triggered when subdirectories whose names are
        **pure integers** (``0``, ``1``, ``2``, ...) are found inside
        ``audio_dir``.  The integers must be contiguous starting from 0 and
        cover every speaker slot — otherwise a ``ValueError`` is raised.

        Legacy names like ``0_Alice`` / ``1_Bob`` are also accepted; the
        integer prefix before the first ``_`` is used as the speaker ID.

        Single-speaker mode is used when no numeric subdirectories are found;
        all audio files are assigned ``default_spk``.
        """
        entries = []

        # ── Discover numeric speaker subdirectories ───────────────────────
        # Accept both "0" (pure int) and "0_Name" (int prefix before _)
        numeric_subdirs: dict[int, Path] = {}
        for d in audio_dir.iterdir():
            if not d.is_dir():
                continue
            raw = d.name.split("_")[0]
            if raw.isdigit():
                sid = int(raw)
                numeric_subdirs[sid] = d

        if numeric_subdirs:
            # ── Multi-speaker validation ──────────────────────────────────
            n = len(numeric_subdirs)
            expected = set(range(n))
            detected = set(numeric_subdirs.keys())

            if detected != expected:
                missing = sorted(expected - detected)
                extra = sorted(detected - expected)
                msg = (
                    f"Multi-speaker folder validation failed in '{audio_dir}'.\n"
                    f"  Expected speaker IDs : {sorted(expected)}\n"
                    f"  Detected speaker IDs : {sorted(detected)}\n"
                )
                if missing:
                    msg += f"  Missing IDs          : {missing}\n"
                if extra:
                    msg += f"  Unexpected IDs       : {extra}\n"
                msg += (
                    "  Folders must be named with contiguous integers starting "
                    "from 0: '0/', '1/', '2/', etc.  "
                    "Optionally followed by a name: '0_Alice/', '1_Bob/', etc."
                )
                raise ValueError(msg)

            logger.info(
                f"Multi-speaker mode: {n} speakers detected "
                f"(IDs {sorted(detected)}) in '{audio_dir}'"
            )

            for sid in sorted(numeric_subdirs):
                subdir = numeric_subdirs[sid]
                for ext in extensions:
                    for f in sorted(subdir.rglob(f"*{ext}")):
                        entries.append((f, sid))
        else:
            # ── Single-speaker mode (or flat sliced_audios/ output) ───────
            # Files produced by Stage 1 (audio.py PreProcess) are already
            # named ``{sid}_{idx0}_{idx1}.wav``.  Detect the sid from the
            # filename prefix so that multi-speaker datasets are handled
            # correctly even when all slices live in a single flat directory.
            for ext in extensions:
                for f in sorted(audio_dir.rglob(f"*{ext}")):
                    raw = f.stem.split("_")[0]
                    sid = int(raw) if raw.isdigit() else default_spk
                    entries.append((f, sid))

        return entries

    @staticmethod
    def _make_stem(audio_path: Path, spk_id: int) -> str:
        """Return the stem to use for output feature files.

        When the input already came from ``sliced_audios/`` (Stage 1 output),
        the filename is already ``{sid}_{idx0}_{idx1}.wav`` — the speaker ID
        is baked into the stem.  Adding another ``{spk_id}_`` prefix would
        produce ``0_0_0_0.npy`` while the GT wav stays ``0_0_0.wav``,
        breaking the intersection in ``generate_filelist``.

        Rule: if the stem already starts with ``{spk_id}_`` (i.e. it was
        produced by Stage 1), use it as-is; otherwise prepend ``{spk_id}_``.
        """
        stem = audio_path.stem
        prefix = f"{spk_id}_"
        if stem.startswith(prefix):
            return stem
        return f"{prefix}{stem}"

    def _generate_filelist(self, output_dir: Path, entries, speaker_ids):
        """
        Generate filelist by delegating to preparing_files.generate_filelist.

        This ensures mute entries are appended and model_info.json is written
        consistently whether called from the pipeline or from the CLI.
        """
        from kazeflow.preprocess.preparing_files import generate_filelist
        generate_filelist(
            model_path=str(output_dir),
            sample_rate=self.sample_rate,
            include_mutes=2,
        )

    def _process_single(self, audio_path: Path, output_dir: Path,
                        speaker_id: int, stem: str = None) -> bool:
        """Process a single audio file. Returns True on success."""
        if stem is None:
            stem = self._make_stem(audio_path, speaker_id)

        # Skip files already processed (all 3 features must exist)
        mel_path = output_dir / "mel" / f"{stem}.npy"
        spin_path = output_dir / "spin" / f"{stem}.npy"
        f0_path = output_dir / "f0" / f"{stem}.npy"
        if mel_path.exists() and spin_path.exists() and f0_path.exists():
            return True

        # Load audio
        audio, sr = torchaudio.load(str(audio_path))
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample to target SR
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        # Duration check
        duration = audio.shape[1] / self.sample_rate
        if duration < self.min_duration:
            logger.debug(f"Skipping {stem}: too short ({duration:.1f}s)")
            return False
        if duration > self.max_duration:
            # Truncate to max duration
            max_samples = int(self.max_duration * self.sample_rate)
            audio = audio[:, :max_samples]

        # Highpass filter
        if self.highpass_freq > 0:
            audio = highpass_biquad(audio, self.sample_rate, self.highpass_freq)

        # Loudness normalization (simple RMS-based)
        audio = self._normalize_loudness(audio)

        audio = audio.to(self.device)

        # ── Extract mel ──────────────────────────────────────────────
        mel = self._compute_mel(audio)  # (n_mels, T)

        # ── Extract SPIN v2 features ─────────────────────────────────
        # SPIN v2 and RMVPE both need 16kHz — resample once, reuse.
        audio_16k = torchaudio.functional.resample(
            audio, self.sample_rate, 16000)
        spin_features = self._extract_spin(audio_16k)  # (768, T_spin)

        # SPIN v2 runs at 50Hz (20ms hop @ 16kHz) while mel runs at 100Hz
        # (10ms hop).  Interpolate SPIN to mel frame rate so features are
        # temporally aligned and no frames are lost during training.
        if spin_features.shape[1] < mel.shape[1]:
            spin_features = F.interpolate(
                spin_features.unsqueeze(0), size=mel.shape[1],
                mode="linear", align_corners=False,
            ).squeeze(0)

        # ── Extract F0 ──────────────────────────────────────────────
        f0 = self._extract_f0(audio_16k.squeeze(0))  # (T_f0,)

        # ── Align lengths ────────────────────────────────────────────
        min_len = min(mel.shape[1], spin_features.shape[1], f0.shape[0])
        mel = mel[:, :min_len]
        spin_features = spin_features[:, :min_len]
        f0 = f0[:min_len]

        # ── Energy-gate SPIN embeddings ──────────────────────────────
        # Zero out SPIN features in frames where the mel energy is near
        # the silence floor.  This teaches the CFM that silence regions
        # have null content, preventing it from hallucinating breathing
        # or tongue artefacts at inference time.
        mel_energy = mel.mean(dim=0)  # (T,) mean across mel channels
        silence_floor = float(torch.tensor(1e-5).log().item())  # ≈ -11.5
        # Soft gate: 0 near floor, 1 for voiced frames
        energy_above_floor = mel_energy - silence_floor  # positive = above floor
        gate = torch.sigmoid(energy_above_floor - 1.0)  # centered ~1 dB above floor
        spin_features = spin_features * gate.unsqueeze(0)  # (C, T) * (1, T)

        # ── Save ─────────────────────────────────────────────────────
        np.save(output_dir / "mel" / f"{stem}.npy", mel.cpu().numpy())
        np.save(output_dir / "spin" / f"{stem}.npy",
                spin_features.cpu().numpy().T)  # Save as (T, 768)
        np.save(output_dir / "f0" / f"{stem}.npy", f0.cpu().numpy())

        # Free GPU tensors promptly (matters for large datasets)
        del mel, spin_features, f0, audio, audio_16k

        return True

    def _normalize_loudness(self, audio: torch.Tensor) -> torch.Tensor:
        """Simple RMS loudness normalization."""
        rms = audio.pow(2).mean().sqrt()
        if rms < 1e-6:
            return audio
        target_rms = 10 ** (self.target_db / 20)
        return audio * (target_rms / rms)

    @torch.no_grad()
    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log mel-spectrogram."""
        # Normalise to 1-D waveform
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        if audio.dim() != 1:
            audio = audio.reshape(-1)
        audio = audio.float()

        window = torch.hann_window(self.win_length, device=audio.device)
        pad = (self.n_fft - self.hop_length) // 2
        # F.pad with mode="reflect" requires at least 2-D input
        audio_padded = F.pad(audio.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
        spec = torch.stft(
            audio_padded.unsqueeze(0),
            n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            center=False, return_complex=True,
        )
        mag = spec.abs().squeeze(0)  # (n_fft//2+1, T)

        mel_basis = self._get_mel_basis()
        mel = torch.matmul(mel_basis, mag)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel

    @torch.no_grad()
    def _extract_spin(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Extract content features from 16kHz audio (SPIN v2 or RSPIN)."""
        model = self._get_spin_model()
        features = model(audio_16k)  # (1, T, embed_dim)
        return features.squeeze(0).T  # (embed_dim, T)

    def _get_rmvpe(self):
        """Lazy-load and cache the RMVPE predictor."""
        if self._rmvpe_model is None:
            from kazeflow.models.predictors.rmvpe import RMVPE0Predictor
            model_path = (
                Path(__file__).parent.parent / "models" / "pretrained"
                / "predictors" / "rmvpe.pt"
            )
            if not model_path.exists():
                raise FileNotFoundError(
                    f"RMVPE weights not found at {model_path}. "
                    "Run app.py to trigger the automatic prerequisites download."
                )
            self._rmvpe_model = RMVPE0Predictor(
                str(model_path), device=self.device
            )
        return self._rmvpe_model

    @torch.no_grad()
    def _extract_f0(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Extract F0 using RMVPE.

        Args:
            audio_16k: mono float32 tensor at 16 kHz (already resampled).
        Returns:
            f0: (T,) float32 tensor on self.device.
        """
        audio_np = audio_16k.cpu().float().numpy()

        rmvpe = self._get_rmvpe()
        # infer_from_audio returns numpy (T,) in Hz; 0 = unvoiced
        f0_np = rmvpe.infer_from_audio(audio_np, thred=0.03)
        f0_np = np.nan_to_num(f0_np, nan=0.0)
        return torch.from_numpy(f0_np).float().to(self.device)
