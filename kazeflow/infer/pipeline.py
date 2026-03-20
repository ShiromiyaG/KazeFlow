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

from kazeflow.models.flow_matching import ConditionalFlowMatching
from kazeflow.models.vocoder import ChouwaGANGenerator
from kazeflow.infer.index import load_index, retrieve_and_blend

logger = logging.getLogger("kazeflow.infer")


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
        self.flow = ConditionalFlowMatching(
            **model_cfg["flow_matching"]
        ).to(self.device).eval()

        self.vocoder = ChouwaGANGenerator(
            sr=model_cfg["sample_rate"],
            **model_cfg["vocoder"],
        ).to(self.device).eval()
        self.vocoder.remove_weight_norm()

        # Speaker embedding
        n_speakers = model_cfg.get("n_speakers", 1)
        spk_dim = model_cfg.get("speaker_embed_dim", 256)
        self.speaker_embed = torch.nn.Embedding(
            n_speakers, spk_dim
        ).to(self.device)

        # Load weights
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.flow.load_state_dict(ckpt["flow"])

        # Prefer EMA vocoder weights for inference (higher quality)
        if "vocoder_ema" in ckpt:
            self.vocoder.load_state_dict(ckpt["vocoder_ema"])
            logger.info("Loaded EMA vocoder weights for inference")
        else:
            self.vocoder.load_state_dict(ckpt["vocoder"])
            logger.info("No EMA weights found — using raw vocoder weights")

        self.speaker_embed.load_state_dict(ckpt["speaker_embed"])

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

    # ── Feature Extraction ────────────────────────────────────────────────

    def _get_spin_model(self):
        """Lazy-load SPIN v2 (HuBERT) model from local pretrained directory."""
        if self._spin_model is None:
            from transformers import HubertModel
            # Prefer the locally downloaded copy; fall back to HF Hub identifier
            # so the first-run download (via prerequisites_download) is the norm.
            local_spin = (
                Path(__file__).parent.parent
                / "models" / "pretrained" / "embedders" / "spin_v2"
            )
            spin_source = str(local_spin) if local_spin.exists() else \
                self.config.get("preprocess", {}).get("spin_model", "dr87/spinv2_rvc")
            self._spin_model = HubertModel.from_pretrained(
                spin_source
            ).to(self.device).eval()
            for p in self._spin_model.parameters():
                p.requires_grad_(False)
        return self._spin_model

    def _extract_spin(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract SPIN v2 features from 16kHz audio.
        Args:
            audio: (1, T) at 16kHz
        Returns:
            features: (1, 768, T_frames)
        """
        model = self._get_spin_model()
        with torch.no_grad():
            outputs = model(audio)
            # Use last hidden state
            features = outputs.last_hidden_state  # (1, T_frames, 768)
        return features.transpose(1, 2)  # (1, 768, T_frames)

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

        # Resample to 16kHz for SPIN v2
        if sr != 16000:
            audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        else:
            audio_16k = audio

        # Resample to model sample rate for F0
        if sr != self.sample_rate:
            audio_model_sr = torchaudio.functional.resample(
                audio, sr, self.sample_rate)
        else:
            audio_model_sr = audio

        # Extract features
        audio_16k = audio_16k.to(self.device)
        spin_features = self._extract_spin(audio_16k)  # (1, 768, T_spin)

        # FAISS index retrieval (blend source features with target speaker)
        if self.faiss_index is not None and index_rate > 0.0:
            spin_features = retrieve_and_blend(
                spin_features, self.faiss_index,
                index_rate=index_rate, k=8, temperature=0.25,
            )

        f0 = self._extract_f0(
            audio_model_sr.squeeze(0), self.sample_rate, method=f0_method
        )  # (1, 1, T_f0)

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

        # Flow matching: generate mel from content + f0 + speaker
        mel_hat = self.flow.sample(
            content=spin_features,
            f0=f0,
            x_mask=x_mask,
            g=g,
            n_steps=ode_steps,
            method=ode_method,
            guidance_scale=guidance_scale,
        )

        # Vocoder: mel → waveform
        f0_squeezed = f0.squeeze(1)  # (1, T)
        waveform = self.vocoder(mel_hat, f0_squeezed, g=g)

        return waveform.squeeze()  # (T_audio,)

    def save_audio(self, waveform: torch.Tensor, output_path: str):
        """Save waveform to file."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(output_path, waveform.cpu(), self.sample_rate)
