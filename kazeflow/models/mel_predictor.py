"""
Direct Mel Regression for voice conversion.

Alternative to CFM: a simple encoder-decoder that directly predicts
mel-spectrograms from content features, F0, and speaker embedding.

No ODE, no noise, no timestep — single forward pass at both train
and inference time.  Deterministic output: same input always produces
the same mel.

Architecture: reuses the same GatedDSConv blocks and AdaGN from the
flow matching module, but without timestep conditioning.  The speaker
embedding alone drives AdaGN modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpeakerAdaGN(nn.Module):
    """
    Adaptive Group Norm conditioned only on speaker embedding.
    (No timestep — this is the key architectural difference from CFM.)
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Linear(cond_dim, channels * 2)
        self.proj.weight.data.zero_()
        self.proj.bias.data[:channels] = 1.0  # scale = 1
        self.proj.bias.data[channels:] = 0.0  # shift = 0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return h * scale[:, :, None] + shift[:, :, None]


class GatedDSConvBlockDM(nn.Module):
    """
    Gated depthwise separable conv block with speaker-only AdaGN.
    Same structure as the CFM version but without timestep.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size * dilation - dilation) // 2

        self.adagn = SpeakerAdaGN(channels, cond_dim)
        self.dwconv = nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=padding, groups=channels,
        )
        self.pwconv1 = nn.Conv1d(channels, channels * 2, 1)
        self.pwconv2 = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        h = self.adagn(x, cond)
        h = self.dwconv(h)
        h = self.pwconv1(h)
        h_a, h_b = h.chunk(2, dim=1)
        h = torch.tanh(h_a) * torch.sigmoid(h_b)
        h = self.drop(h)
        skip = self.skip_proj(h) * x_mask
        h = self.pwconv2(h)
        return (residual + h) * x_mask, skip


class F0Conditioning(nn.Module):
    """
    Dedicated F0 feature extraction.

    Processes F0 through a separate path before combining with content,
    treating voiced/unvoiced explicitly.  This avoids diluting the F0
    signal (1 channel) when concatenated against dense content features
    (768 channels).
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        half = hidden_channels // 2
        self.voiced_proj = nn.Conv1d(1, half, 1)
        self.uv_embed = nn.Embedding(2, half)  # 0 = unvoiced, 1 = voiced
        self.out_proj = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        # f0: (B, 1, T)
        voiced = (f0 > 0).long().squeeze(1)            # (B, T)
        uv = self.uv_embed(voiced).transpose(1, 2)     # (B, H/2, T)
        f0_cont = self.voiced_proj(f0.clamp(min=0))    # (B, H/2, T)
        return self.out_proj(torch.cat([f0_cont, uv], dim=1))


class Snake(nn.Module):
    """
    Periodic activation: x + (1/a) * sin²(ax).

    Developed for audio generation (BigVGAN).  Has inductive bias for
    harmonic structure — better than generic SiLU for mel output.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.clamp(min=1e-5)
        return x + (1.0 / a) * torch.sin(a * x) ** 2


class MelEstimator(nn.Module):
    """
    Direct mel-spectrogram estimator.

    Predicts mel = f(content, f0, speaker) in a single forward pass.
    No noisy input x_t, no timestep t.

    Architecture: stack of GatedDSConvBlocks with speaker-only AdaGN.
    F0 is embedded through a dedicated path (voiced/unvoiced aware).
    Output uses Snake activation for harmonic bias.
    Skip connections use learned per-layer scaling.
    """

    def __init__(
        self,
        mel_channels: int,
        hidden_channels: int,
        cond_channels: int,
        gin_channels: int,
        n_layers: int = 12,
        kernel_size: int = 7,
        dilation_rate: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.mel_channels = mel_channels

        # Speaker projection -> conditioning dimension for AdaGN
        spk_dim = hidden_channels * 2
        if gin_channels > 0:
            self.spk_proj = nn.Linear(gin_channels, spk_dim)
        else:
            self.spk_proj = None
            spk_dim = hidden_channels * 2  # fallback

        # Dedicated F0 conditioning (voiced/unvoiced aware)
        self.f0_cond = F0Conditioning(hidden_channels)

        # Content projection: content features -> hidden (no F0 concat)
        self.cond_proj = nn.Conv1d(cond_channels, hidden_channels, 1)

        # Gated DS conv blocks with exponential dilations
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** (i % 8)
            self.blocks.append(GatedDSConvBlockDM(
                hidden_channels, spk_dim, kernel_size, dilation, dropout,
            ))

        # Adaptive skip scaling: learned per-layer importance
        self.skip_scale = nn.Parameter(torch.ones(n_layers))

        # Output projection: hidden -> mel
        self.out_norm = nn.GroupNorm(1, hidden_channels)
        self.snake = Snake(hidden_channels)
        self.out_proj = nn.Conv1d(hidden_channels, mel_channels, 1)
        self.out_proj.weight.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(
        self,
        content: torch.Tensor,     # (B, cond_ch, T)
        f0: torch.Tensor,          # (B, 1, T)
        x_mask: torch.Tensor,      # (B, 1, T)
        g: Optional[torch.Tensor] = None,  # (B, gin_ch, 1) speaker
    ) -> torch.Tensor:
        # Speaker conditioning -> (B, spk_dim)
        if g is not None and self.spk_proj is not None:
            spk_cond = self.spk_proj(g.squeeze(-1))
        else:
            spk_cond = torch.zeros(
                content.size(0), self.hidden_channels * 2,
                device=content.device, dtype=content.dtype,
            )

        # Content + dedicated F0 embedding (additive, not concat)
        f0_feat = self.f0_cond(f0)                     # (B, H, T)
        h = self.cond_proj(content) + f0_feat           # (B, H, T)
        h = h * x_mask

        # Gated DS conv stack with adaptive skip accumulation
        skip_sum = torch.zeros_like(h)
        for i, block in enumerate(self.blocks):
            h, skip = block(h, spk_cond, x_mask)
            skip_sum = skip_sum + self.skip_scale[i] * skip

        # Normalize by L2 norm of skip scales (not fixed sqrt(N))
        norm = self.skip_scale.norm().clamp(min=1e-5)
        h = skip_sum / norm
        h = self.out_norm(h)
        h = self.snake(h)
        mel = self.out_proj(h) * x_mask
        return mel


class DirectMelPredictor(nn.Module):
    """
    Direct Mel Regression module for voice conversion.

    Training: predict mel directly from conditioning, loss = L1(mel_hat, mel_gt).
    Inference: single forward pass — deterministic, fast, no ODE.

    Drop-in alternative to ConditionalFlowMatching with the same interface
    for conditioning (content, f0, speaker embedding).
    """

    def __init__(
        self,
        mel_channels: int = 128,
        hidden_channels: int = 256,
        cond_channels: int = 768,
        gin_channels: int = 256,
        n_layers: int = 12,
        kernel_size: int = 7,
        dilation_rate: int = 2,
        dropout: float = 0.05,
        # Unused CFM params — accepted for config compatibility
        sigma_min: float = 1e-4,
        t_sampling: str = "logit_normal",
        t_logit_mean: float = 0.0,
        t_logit_std: float = 1.0,
        cfg_dropout: float = 0.0,
    ):
        super().__init__()
        self.estimator = MelEstimator(
            mel_channels=mel_channels,
            hidden_channels=hidden_channels,
            cond_channels=cond_channels,
            gin_channels=gin_channels,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=dropout,
        )

    def predict(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return mel prediction WITH gradients (for adversarial training)."""
        return self.estimator(content, f0, x_mask, g)

    def forward(
        self,
        x_1: torch.Tensor,         # (B, mel_ch, T) target mel
        x_mask: torch.Tensor,       # (B, 1, T)
        content: torch.Tensor,      # (B, cond_ch, T)
        f0: torch.Tensor,           # (B, 1, T)
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training forward: compute L1 mel prediction loss."""
        mel_hat = self.estimator(content, f0, x_mask, g)

        # L1 loss (masked)
        C = x_1.shape[1]
        error = (mel_hat - x_1) * x_mask
        loss = error.abs().sum() / (x_mask.sum() * C)
        return loss

    @torch.no_grad()
    def sample(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        # Unused ODE params — accepted for interface compatibility
        n_steps: int = 16,
        method: str = "euler",
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate mel-spectrogram — single forward pass.

        Accepts the same arguments as ConditionalFlowMatching.sample()
        for drop-in compatibility.  n_steps, method, and guidance_scale
        are ignored (no ODE, deterministic output).
        """
        return self.estimator(content, f0, x_mask, g)
