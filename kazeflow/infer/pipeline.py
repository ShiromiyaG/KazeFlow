"""
KazeFlow Inference Pipeline.

Full voice conversion pipeline:
1. Extract SPIN v2 features from source audio
2. (Optional) Retrieve and blend with FAISS index for speaker similarity
3. Extract F0 from source audio (with optional pitch shift)
4. Generate mel-spectrogram via flow matching ODE
5. Synthesize waveform via ChouwaGAN vocoder
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from kazeflow.models import build_mel_model
from kazeflow.models.vocoder import build_vocoder
from kazeflow.infer.index import load_index, retrieve_and_blend

logger = logging.getLogger("kazeflow.infer")


def _deparametrize_state_dict(sd: dict) -> dict:
    """Convert a parametrized weight-norm state dict to plain weights.

    During training, ``torch.nn.utils.parametrizations.weight_norm`` stores
    each weight as ``parametrizations.weight.original0`` (magnitude) and
    ``parametrizations.weight.original1`` (direction).  At inference time
    ``remove_weight_norm()`` collapses them back to a single ``weight``
    tensor.  This function performs the same collapse on a state dict so
    it can be loaded into a model that already had weight norm removed.

    Also strips ``_orig_mod.`` prefixes from ``torch.compile``.
    """
    # Strip _orig_mod. prefixes (torch.compile)
    if sd and all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}

    # Collect parametrized weight groups
    # Keys look like: "head.conv_pre.parametrizations.weight.original0"
    #                  "head.conv_pre.parametrizations.weight.original1"
    param_groups: dict[str, dict[str, torch.Tensor]] = {}
    plain = {}
    for k, v in sd.items():
        if ".parametrizations.weight.original" in k:
            # "head.conv_pre.parametrizations.weight.original0" -> prefix="head.conv_pre", idx="0"
            prefix = k.split(".parametrizations.weight.original")[0]
            idx = k.split(".parametrizations.weight.original")[1]
            param_groups.setdefault(prefix, {})[idx] = v
        else:
            plain[k] = v

    # Collapse: weight = original0 * (original1 / ||original1||)
    for prefix, parts in param_groups.items():
        if "0" in parts and "1" in parts:
            g = parts["0"]   # magnitude (scalar per output channel)
            v = parts["1"]   # direction (full weight tensor)
            # Normalize v over all dims except dim 0
            dims = list(range(1, v.dim()))
            norm = v.norm(2, dim=dims, keepdim=True).clamp(min=1e-12)
            plain[f"{prefix}.weight"] = g * (v / norm)
        elif "0" in parts:
            plain[f"{prefix}.weight"] = parts["0"]
        elif "1" in parts:
            plain[f"{prefix}.weight"] = parts["1"]

    return plain


class KazeFlowPipeline:
    """End-to-end voice conversion inference."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        index_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)

        # Load config
        if config_path is None:
            config_path = str(Path(checkpoint_path).parent / "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        model_cfg = self.config["model"]

        # Build models
        self.architecture = model_cfg.get("architecture", "cfm")
        self.flow = build_mel_model(
            self.architecture, **model_cfg["flow_matching"]
        ).to(self.device).eval()

        vocoder_type = model_cfg.get("vocoder_type", "chouwa_gan")
        self.vocoder = build_vocoder(
            vocoder_type,
            sr=model_cfg["sample_rate"],
            **model_cfg["vocoder"],
        ).to(self.device).eval()
        # NOTE: Do NOT remove_weight_norm yet — the EMA state dict stores
        # parametrized keys (original0/1).  We load weights first, then
        # remove WN so PyTorch collapses them natively (no manual error).

        # Speaker embedding
        n_speakers = model_cfg.get("n_speakers", 1)
        spk_dim = model_cfg.get("speaker_embed_dim", 256)
        self.speaker_embed = torch.nn.Embedding(
            n_speakers, spk_dim
        ).to(self.device)

        # Load weights — prefer EMA shadows (higher quality, matches eval)
        # Fall back to raw training weights for old checkpoints without EMA
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        _flow_sd = ckpt.get("flow_ema", ckpt["flow"])
        _voc_sd = ckpt.get("vocoder_ema", ckpt["vocoder"])
        self.flow.load_state_dict(_deparametrize_state_dict(_flow_sd))

        # Vocoder loading: the EMA state dict has weight_norm parametrized
        # keys (original0/original1).  Load into the model WITH weight_norm
        # intact so PyTorch matches the parametrizations natively, then
        # remove_weight_norm() collapses them.
        _load_result = self.vocoder.load_state_dict(_voc_sd, strict=False)
        if _load_result.missing_keys:
            # Weight-norm keys won't match if checkpoint has plain weights
            # (old format). Try deparametrize + reload.
            _dp_sd = _deparametrize_state_dict(_voc_sd)
            # Must remove WN first so model expects plain keys
            self.vocoder.remove_weight_norm()
            _load_result = self.vocoder.load_state_dict(_dp_sd, strict=False)
            _wn_removed = True
        else:
            # Loaded successfully with WN keys — remove WN natively
            self.vocoder.remove_weight_norm()
            _wn_removed = False

        if _load_result.missing_keys:
            logger.warning(
                "Vocoder missing keys (%d): %s",
                len(_load_result.missing_keys),
                _load_result.missing_keys[:10],
            )
        if _load_result.unexpected_keys:
            logger.warning(
                "Vocoder unexpected keys (%d): %s",
                len(_load_result.unexpected_keys),
                _load_result.unexpected_keys[:10],
            )

        self.speaker_embed.load_state_dict(
            _deparametrize_state_dict(ckpt["speaker_embed"]))
        _ema_label = "EMA" if "flow_ema" in ckpt else "raw (no EMA in checkpoint)"
        logger.info(f"Loaded flow + vocoder ({_ema_label}) + speaker_embed from checkpoint")

        # FAISS index (optional)
        self.faiss_index = None
        if index_path is not None:
            self.faiss_index = load_index(index_path)

        # Lazy-loaded feature extractors
        self._spin_model = None
        self._f0_extractor = None
        self._rmvpe_model = None

        self.sample_rate = model_cfg["sample_rate"]
        self.hop_length = model_cfg["hop_length"]
        self.n_fft = model_cfg["n_fft"]
        self.win_length = model_cfg["win_length"]
        self.n_mels = model_cfg["n_mels"]
        self._mel_basis = None

    # ── Feature Extraction ────────────────────────────────────────────────

    def _get_mel_basis(self):
        """Lazy-init mel filterbank (matches preprocessing pipeline)."""
        if self._mel_basis is None:
            self._mel_basis = torchaudio.functional.melscale_fbanks(
                n_freqs=self.n_fft // 2 + 1,
                f_min=0.0, f_max=self.sample_rate / 2.0,
                n_mels=self.n_mels, sample_rate=self.sample_rate,
            ).T.to(self.device)  # (n_mels, n_fft//2+1)
        return self._mel_basis

    @torch.no_grad()
    def _compute_source_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log mel-spectrogram from source audio (for energy gating).

        Matches the preprocessing pipeline's _compute_mel exactly.
        """
        if audio.dim() > 1:
            audio = audio.squeeze(0)
        audio = audio.float()

        window = torch.hann_window(self.win_length, device=audio.device)
        pad = (self.n_fft - self.hop_length) // 2
        audio_padded = F.pad(
            audio.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
        spec = torch.stft(
            audio_padded.unsqueeze(0),
            n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            center=False, return_complex=True,
        )
        mag = spec.abs()  # (1, n_fft//2+1, T)

        mel_basis = self._get_mel_basis()
        mel = torch.matmul(mel_basis, mag.squeeze(0))  # (n_mels, T)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel.unsqueeze(0)  # (1, n_mels, T)

    def _get_spin_model(self):
        """Lazy-load content embedder (SPIN v2 or RSPIN)."""
        if self._spin_model is None:
            from kazeflow.models.embedder import load_content_embedder
            embedder_name = self.config.get("preprocess", {}).get(
                "content_embedder", "spin_v2")
            spin_source = self.config.get("preprocess", {}).get(
                "spin_model", "dr87/spinv2_rvc")
            self._spin_model = load_content_embedder(
                name=embedder_name,
                device=str(self.device),
                spin_source=spin_source,
            )
        return self._spin_model

    def _extract_spin(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract content features from 16kHz audio (SPIN v2 or RSPIN).
        Processes in chunks to avoid OOM on long audio.
        Args:
            audio: (1, T) at 16kHz
        Returns:
            features: (1, embed_dim, T_frames)
        """
        model = self._get_spin_model()
        # Ensure model is on the right device (may have been offloaded to CPU)
        model.to(self.device)
        # ~30s chunks at 16kHz with 1s overlap for smooth transitions
        chunk_samples = 30 * 16000   # 480000 samples
        overlap_samples = 1 * 16000  # 16000 samples
        T = audio.shape[1]

        if T <= chunk_samples:
            with torch.no_grad():
                features = model(audio)  # (1, T_frames, embed_dim)
            return features.transpose(1, 2)

        # Chunked extraction
        stride = chunk_samples - overlap_samples
        chunks_out = []
        start = 0
        while start < T:
            end = min(start + chunk_samples, T)
            chunk = audio[:, start:end]
            with torch.no_grad():
                feat = model(chunk)  # (1, T_frames, embed_dim)
            # For overlapping regions, only keep the non-overlapping part
            # except for the first chunk where we keep everything,
            # and for subsequent chunks we skip the overlap frames
            if start == 0:
                chunks_out.append(feat)
            else:
                # Estimate how many frames correspond to the overlap
                # Frames = samples / hop (16kHz / 320 = 50 Hz for SPIN)
                overlap_frames = overlap_samples // 320
                if overlap_frames < feat.shape[1]:
                    chunks_out.append(feat[:, overlap_frames:, :])
                else:
                    chunks_out.append(feat)
            start += stride
            if end == T:
                break

        features = torch.cat(chunks_out, dim=1)  # (1, total_frames, embed_dim)
        return features.transpose(1, 2)

    def _extract_f0(self, audio: torch.Tensor, sr: int,
                    method: str = "rmvpe") -> torch.Tensor:
        """
        Extract F0 contour.
        Args:
            audio: (T,) numpy array or tensor at original sr
            method: "rmvpe" (only supported method)
        Returns:
            f0: (1, 1, T_frames) in Hz
        """
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        if method == "rmvpe":
            f0 = self._extract_f0_rmvpe(audio_np, sr)
        else:
            raise ValueError(f"Unknown F0 method: {method}")

        f0 = torch.from_numpy(f0).float().to(self.device)
        return f0.unsqueeze(0).unsqueeze(0)  # (1, 1, T_frames)

    def _get_rmvpe(self):
        """Lazy-load and cache the RMVPE0Predictor."""
        if self._rmvpe_model is None:
            from kazeflow.models.predictors.rmvpe import RMVPE0Predictor
            weights = (
                Path(__file__).parent.parent
                / "models" / "pretrained" / "predictors" / "rmvpe.pt"
            )
            if not weights.exists():
                raise FileNotFoundError(
                    f"RMVPE weights not found at {weights}. "
                    "Run app.py to trigger the automatic prerequisites download."
                )
            self._rmvpe_model = RMVPE0Predictor(str(weights), device=self.device)
        else:
            # Restore to GPU if previously offloaded
            self._rmvpe_model.model.to(self.device)
            self._rmvpe_model.device = self.device
        return self._rmvpe_model

    def _extract_f0_rmvpe(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """RMVPE F0 extraction. Expects mono numpy array at any sample rate."""
        # RMVPE requires 16 kHz input
        if sr != 16000:
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, 16000)
            audio_16k = audio_t.squeeze(0).numpy()
        else:
            audio_16k = audio

        rmvpe = self._get_rmvpe()
        f0 = rmvpe.infer_from_audio(audio_16k, thred=0.03)  # numpy (T,) Hz
        return f0

    # ── Main Inference ────────────────────────────────────────────────────

    @torch.no_grad()
    def convert(
        self,
        source_audio_path: str,
        speaker_id: int = 0,
        f0_shift: int = 0,
        ode_steps: int = 16,
        ode_method: str = "euler",
        f0_method: str = "rmvpe",
        index_rate: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Voice conversion: source audio → target speaker waveform.

        Args:
            source_audio_path: Path to source audio file
            speaker_id: Target speaker index
            f0_shift: Pitch shift in semitones
            ode_steps: Number of ODE solver steps
            ode_method: "euler" or "midpoint"
            f0_method: F0 extraction method
            index_rate: FAISS index blending (0.0=off, 1.0=full retrieval)
            guidance_scale: Classifier-Free Guidance scale. 1.0 = no guidance,
                >1.0 = amplify speaker conditioning (requires cfg_dropout > 0
                during training). Typical range: 1.0-3.0.

        Returns:
            waveform: (T_audio,) tensor at model sample rate
        """
        # Load audio
        audio, sr = torchaudio.load(source_audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # ── Preprocessing: match training data pipeline ──────────────
        # The preprocessing pipeline applies highpass filtering, loudness
        # normalization, and energy-gating before saving features.  We
        # must replicate these steps here so the SPIN/F0 features match
        # what the flow model was trained on.

        # Resample to model sample rate first (preprocessing works at
        # model sr, then derives 16kHz from that)
        if sr != self.sample_rate:
            audio_model_sr = torchaudio.functional.resample(
                audio, sr, self.sample_rate)
        else:
            audio_model_sr = audio.clone()

        # Highpass filter (remove DC / low-frequency rumble)
        preproc_cfg = self.config.get("preprocess", {})
        hp_freq = preproc_cfg.get("highpass_freq", 48)
        if hp_freq > 0:
            audio_model_sr = torchaudio.functional.highpass_biquad(
                audio_model_sr, self.sample_rate, hp_freq)

        # Loudness normalization (simple RMS-based, matches preprocessing)
        target_db = preproc_cfg.get("target_db", -23.0)
        rms = audio_model_sr.pow(2).mean().sqrt()
        if rms > 1e-6:
            target_rms = 10 ** (target_db / 20)
            audio_model_sr = audio_model_sr * (target_rms / rms)

        # Derive 16kHz from the preprocessed audio (same path as training)
        audio_16k = torchaudio.functional.resample(
            audio_model_sr, self.sample_rate, 16000)

        # Extract features — order matters for VRAM management:
        # 1. F0 (small RMVPE model)
        # 2. SPIN (large WavLM backbone)
        # 3. Offload both to CPU before flow + vocoder

        f0 = self._extract_f0(
            audio_model_sr.squeeze(0), self.sample_rate, method=f0_method
        )  # (1, 1, T_f0)

        audio_16k = audio_16k.to(self.device)
        spin_features = self._extract_spin(audio_16k)  # (1, embed_dim, T_spin)

        # ── Energy-gate SPIN features ────────────────────────────────
        # During preprocessing, SPIN features are zeroed in silent frames
        # using a mel-energy gate.  Without this, the flow model sees
        # non-zero SPIN in silence and generates noise where there should
        # be nothing.
        # Compute mel from source audio for gating (matches preprocessing)
        mel_for_gate = self._compute_source_mel(
            audio_model_sr.to(self.device))  # (1, n_mels, T_mel)
        mel_energy = mel_for_gate.mean(dim=1)  # (1, T_mel)

        # Interpolate mel energy to SPIN frame rate
        if mel_energy.shape[1] != spin_features.shape[2]:
            mel_energy = F.interpolate(
                mel_energy.unsqueeze(1),
                size=spin_features.shape[2],
                mode="linear", align_corners=False,
            ).squeeze(1)  # (1, T_spin)

        silence_floor = float(torch.tensor(1e-5).log().item())  # ≈ -11.5
        energy_above_floor = mel_energy - silence_floor
        gate = torch.sigmoid(energy_above_floor - 1.0)  # (1, T_spin)
        spin_features = spin_features * gate.unsqueeze(1)  # (1, C, T) * (1, 1, T)
        logger.info("Energy-gated SPIN: %.1f%% frames silenced (gate<0.1)",
                     100 * (gate < 0.1).float().mean().item())

        del mel_for_gate, mel_energy, gate

        # Free feature extraction models from GPU
        if self._spin_model is not None:
            self._spin_model.cpu()
        if self._rmvpe_model is not None:
            self._rmvpe_model.model.cpu()
        del audio_16k, audio_model_sr
        torch.cuda.empty_cache()

        # FAISS index retrieval (blend source features with target speaker)
        if self.faiss_index is not None and index_rate > 0.0:
            spin_features = retrieve_and_blend(
                spin_features, self.faiss_index,
                index_rate=index_rate, k=8, temperature=0.25,
            )

        # Apply pitch shift (semitones → frequency ratio)
        if f0_shift != 0:
            ratio = 2.0 ** (f0_shift / 12.0)
            f0 = f0 * ratio

        # SPIN v2 runs at 50Hz while F0 runs at 100Hz.  Interpolate SPIN
        # to F0's frame rate so the output mel has full temporal resolution.
        if spin_features.shape[2] < f0.shape[2]:
            spin_features = torch.nn.functional.interpolate(
                spin_features, size=f0.shape[2], mode="linear",
                align_corners=False,
            )

        # Align lengths
        min_len = min(spin_features.shape[2], f0.shape[2])
        spin_features = spin_features[:, :, :min_len]
        f0 = f0[:, :, :min_len]

        # Speaker embedding
        spk_id_tensor = torch.tensor([speaker_id], device=self.device)
        g = self.speaker_embed(spk_id_tensor).unsqueeze(-1)

        # Mask
        x_mask = torch.ones(1, 1, min_len, device=self.device)

        # Generate mel from conditioning
        if self.architecture == "direct_mel":
            # Direct mel regression: single deterministic forward pass
            mel_hat = self.flow.sample(
                content=spin_features,
                f0=f0,
                x_mask=x_mask,
                g=g,
            )
        else:
            # Flow matching: ODE solver
            # NOTE: ODE solver MUST run in float32 — float16 accumulates
            # catastrophic error over multiple integration steps.
            mel_hat = self.flow.sample(
                content=spin_features,
                f0=f0,
                x_mask=x_mask,
                g=g,
                n_steps=ode_steps,
                method=ode_method,
                guidance_scale=guidance_scale,
            )

        # Free flow intermediates before vocoder
        del spin_features, x_mask
        torch.cuda.empty_cache()

        # Diagnostic: log mel_hat statistics to verify flow output quality
        logger.info(
            "mel_hat stats: shape=%s  min=%.3f  max=%.3f  mean=%.3f  std=%.3f",
            list(mel_hat.shape), mel_hat.min().item(), mel_hat.max().item(),
            mel_hat.mean().item(), mel_hat.std().item(),
        )

        # Vocoder: mel → waveform (chunked for long audio to avoid OOM)
        f0_squeezed = f0.squeeze(1)  # (1, T)
        del f0

        _chunk_frames = 400   # ~4s at 100Hz frame rate
        _overlap_frames = 16  # overlap for smooth crossfade
        T_mel = mel_hat.shape[2]

        if T_mel <= _chunk_frames:
            waveform = self.vocoder(mel_hat, f0_squeezed, g=g)
        else:
            _stride = _chunk_frames - _overlap_frames
            _hop = self.hop_length
            _xfade_samples = _overlap_frames * _hop
            wav_chunks = []
            prev_overlap = None  # saved tail from previous chunk

            pos = 0
            while pos < T_mel:
                end = min(pos + _chunk_frames, T_mel)
                mel_chunk = mel_hat[:, :, pos:end]
                f0_chunk = f0_squeezed[:, pos:end]

                wav_chunk = self.vocoder(mel_chunk, f0_chunk, g=g)
                wav_chunk = wav_chunk.float().squeeze(0).squeeze(0)

                if prev_overlap is not None:
                    # Crossfade: blend previous chunk tail with current chunk head
                    cur_head = wav_chunk[:_xfade_samples]
                    fade_in = torch.linspace(0, 1, _xfade_samples, device=wav_chunk.device)
                    blended = prev_overlap * (1 - fade_in) + cur_head * fade_in
                    wav_chunks.append(blended)
                    body_start = _xfade_samples
                else:
                    body_start = 0

                if end < T_mel:
                    # Save tail for crossfade, emit body without tail
                    prev_overlap = wav_chunk[-_xfade_samples:].clone()
                    wav_chunks.append(wav_chunk[body_start:-_xfade_samples])
                else:
                    # Last chunk: emit everything remaining
                    wav_chunks.append(wav_chunk[body_start:])
                    prev_overlap = None

                pos += _stride
                del mel_chunk, f0_chunk, wav_chunk
                torch.cuda.empty_cache()

            waveform = torch.cat(wav_chunks, dim=0).unsqueeze(0)

        wav_out = waveform.float().squeeze()  # (T_audio,)
        logger.info(
            "waveform stats: len=%d  min=%.4f  max=%.4f  abs_mean=%.4f",
            wav_out.shape[0], wav_out.min().item(), wav_out.max().item(),
            wav_out.abs().mean().item(),
        )
        return wav_out

    def save_audio(self, waveform: torch.Tensor, output_path: str):
        """Save waveform to file."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(output_path, waveform.cpu(), self.sample_rate)
