"""
Conditional Flow Matching (CFM) for voice conversion.

Replaces VITS's VAE+Flow with a simple ODE that transports Gaussian noise
to mel-spectrograms conditioned on content features, F0, and speaker embedding.
No KL divergence — the ODE loss directly optimizes the velocity field.

Architecture: DiT-style (Diffusion Transformer) adapted for 1D audio:
- Depthwise separable gated convolutions (same as LightConvNet)
- Sinusoidal timestep conditioning via AdaGN (Adaptive Group Norm)
- FiLM speaker conditioning
- Bidirectional with exponential dilations for large receptive field

Inference: Euler/midpoint ODE solver with configurable number of steps.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from kazeflow.models.commons import sequence_mask


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Adaptive Group Norm (timestep + speaker conditioning)
# ---------------------------------------------------------------------------

class AdaGN(nn.Module):
    """
    Adaptive Group Norm: GroupNorm + affine modulation from timestep/speaker.
    Fuses timestep and speaker conditioning into a single scale+shift,
    avoiding separate FiLM layers for each.
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Linear(cond_dim, channels * 2)
        self.proj.weight.data.zero_()
        self.proj.bias.data[:channels] = 1.0  # scale = 1
        self.proj.bias.data[channels:] = 0.0  # shift = 0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        h = self.norm(x)
        scale_shift = self.proj(cond)  # (B, 2C)
        scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, C)
        return h * scale[:, :, None] + shift[:, :, None]


# ---------------------------------------------------------------------------
# Gated Depthwise Separable Conv Block (with AdaGN conditioning)
# ---------------------------------------------------------------------------

class GatedDSConvBlock(nn.Module):
    """
    Gated depthwise separable convolution block with AdaGN conditioning.

    AdaGN(cond) -> DwConv(dilated) -> PwConv(->2C) -> tanh*sigmoid -> PwConv(->C)
    + residual connection + skip output for WaveNet-style accumulation.
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

        self.adagn = AdaGN(channels, cond_dim)
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
        # Gated activation (tanh * sigmoid)
        h_a, h_b = h.chunk(2, dim=1)
        h = torch.tanh(h_a) * torch.sigmoid(h_b)
        h = self.drop(h)
        skip = self.skip_proj(h) * x_mask
        h = self.pwconv2(h)
        return (residual + h) * x_mask, skip


# ---------------------------------------------------------------------------
# Flow Matching Estimator (velocity field v_θ)
# ---------------------------------------------------------------------------

class FlowEstimator(nn.Module):
    """
    Velocity field estimator for Conditional Flow Matching.

    Predicts v_θ(x_t, t, cond) where:
    - x_t = (1-t)*noise + t*x_1 (interpolated sample)
    - t ∈ [0, 1] (timestep)
    - cond = (content_features, f0, speaker_embedding)

    Architecture: stack of GatedDSConvBlocks with exponential dilations
    and skip accumulation (WaveNet-style multi-scale output).
    """

    def __init__(
        self,
        in_channels: int,        # mel channels (e.g. 128)
        hidden_channels: int,    # internal dimension (e.g. 256)
        cond_channels: int,      # content feature dim (e.g. 768 for SPIN v2)
        gin_channels: int,       # speaker embedding dim (e.g. 256)
        n_layers: int = 12,
        kernel_size: int = 7,
        dilation_rate: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Timestep embedding: t -> time_dim
        time_dim = hidden_channels * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition projection: content features + f0 -> hidden
        # +1 for f0 (continuous voiced frequency)
        self.cond_proj = nn.Conv1d(cond_channels + 1, hidden_channels, 1)

        # Speaker projection -> combined with timestep
        if gin_channels > 0:
            self.spk_proj = nn.Linear(gin_channels, time_dim)
        else:
            self.spk_proj = None

        # Input projection: noisy mel -> hidden
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        # Merge input + condition
        self.merge = nn.Conv1d(hidden_channels * 2, hidden_channels, 1)

        # Gated DS conv blocks with exponential dilations
        cond_dim = time_dim  # AdaGN conditioning dimension
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** (i % 8)  # cycle dilations for deep nets
            self.blocks.append(GatedDSConvBlock(
                hidden_channels, cond_dim, kernel_size, dilation, dropout,
            ))

        # Output projection: hidden -> mel (velocity prediction)
        self.out_norm = nn.GroupNorm(1, hidden_channels)
        self.out_proj = nn.Conv1d(hidden_channels, in_channels, 1)
        self.out_proj.weight.data.zero_()
        self.out_proj.bias.data.zero_()

    def forward(
        self,
        x_t: torch.Tensor,        # (B, mel_ch, T) noisy mel
        t: torch.Tensor,           # (B,) timestep ∈ [0, 1]
        content: torch.Tensor,     # (B, cond_ch, T) content features
        f0: torch.Tensor,          # (B, 1, T) continuous f0
        x_mask: torch.Tensor,      # (B, 1, T) mask
        g: Optional[torch.Tensor] = None,  # (B, gin_ch, 1) speaker
    ) -> torch.Tensor:
        # Timestep + speaker conditioning -> (B, time_dim)
        t_emb = self.time_mlp(t)
        if g is not None and self.spk_proj is not None:
            t_emb = t_emb + self.spk_proj(g.squeeze(-1))

        # Input: noisy mel
        h_in = self.in_proj(x_t) * x_mask

        # Condition: content + f0
        cond = torch.cat([content, f0], dim=1)
        h_cond = self.cond_proj(cond) * x_mask

        # Merge
        h = self.merge(torch.cat([h_in, h_cond], dim=1)) * x_mask

        # Gated DS conv stack with skip accumulation
        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, t_emb, x_mask)
            skip_sum = skip_sum + skip

        # Output from accumulated skips (WaveNet-style)
        h = skip_sum * (1.0 / len(self.blocks) ** 0.5)
        h = self.out_norm(h)
        h = F.silu(h)
        v = self.out_proj(h) * x_mask
        return v


# ---------------------------------------------------------------------------
# Conditional Flow Matching Module (training + inference)
# ---------------------------------------------------------------------------

class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching for voice conversion.

    Training: Sample t ~ U(0,1), interpolate x_t = (1-t)*x_0 + t*x_1,
    predict velocity v_θ(x_t, t, cond), loss = ||v_θ - (x_1 - x_0)||².

    Inference: Solve ODE from x_0 ~ N(0,1) to x_1 using Euler/midpoint.
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
        sigma_min: float = 1e-4,
        t_sampling: str = "logit_normal",
        t_logit_mean: float = 0.0,
        t_logit_std: float = 1.0,
        cfg_dropout: float = 0.0,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.t_sampling = t_sampling
        self.t_logit_mean = t_logit_mean
        self.t_logit_std = t_logit_std
        self.cfg_dropout = cfg_dropout

        self.estimator = FlowEstimator(
            in_channels=mel_channels,
            hidden_channels=hidden_channels,
            cond_channels=cond_channels,
            gin_channels=gin_channels,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=dropout,
        )

    def forward(
        self,
        x_1: torch.Tensor,         # (B, mel_ch, T) target mel
        x_mask: torch.Tensor,       # (B, 1, T)
        content: torch.Tensor,      # (B, cond_ch, T)
        f0: torch.Tensor,           # (B, 1, T)
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training forward: compute CFM loss."""
        B, C, T = x_1.shape

        # Sample timestep
        if self.t_sampling == "logit_normal":
            # Logit-normal distribution (SD3 paper): concentrates near t=0.5
            # where the model learns the most
            u = torch.randn(B, device=x_1.device, dtype=x_1.dtype)
            t = torch.sigmoid(u * self.t_logit_std + self.t_logit_mean)
        else:
            # Uniform sampling (standard CFM)
            t = torch.rand(B, device=x_1.device, dtype=x_1.dtype)
        # Avoid exact 0 and 1 for numerical stability
        t = t.clamp(self.sigma_min, 1.0 - self.sigma_min)

        # Sample noise x_0 ~ N(0, 1)
        x_0 = torch.randn_like(x_1)

        # Optimal transport path with sigma_min:
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        # At t=0: x_t = x_0 (pure noise)
        # At t=1: x_t ≈ x_1 + sigma_min * x_0 (target with residual noise)
        t_expand = t[:, None, None]  # (B, 1, 1)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_expand) * x_0 + t_expand * x_1

        # Target velocity: dx/dt = x_1 - (1 - sigma_min) * x_0
        u_t = x_1 - (1.0 - self.sigma_min) * x_0

        # Classifier-Free Guidance: randomly drop speaker embedding
        # during training so the model learns unconditional generation
        g_train = g
        if g is not None and self.cfg_dropout > 0.0 and self.training:
            drop_mask = torch.rand(B, device=x_1.device) < self.cfg_dropout
            if drop_mask.any():
                g_train = g.clone()
                g_train[drop_mask] = 0.0

        # Predict velocity
        v_t = self.estimator(x_t, t, content, f0, x_mask, g_train)

        # CFM loss: ||v_θ - u_t||² masked, normalized per element (B×C×T)
        loss = F.mse_loss(v_t * x_mask, u_t * x_mask, reduction="sum")
        loss = loss / (x_mask.sum() * C)

        return loss

    def _cfg_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        content: torch.Tensor,
        f0: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Compute velocity with optional Classifier-Free Guidance.

        When guidance_scale > 1.0, runs two forward passes:
          v = v_uncond + guidance_scale * (v_cond - v_uncond)
        This amplifies the speaker-conditioned direction relative to
        the unconditional prediction, improving speaker similarity.

        When guidance_scale <= 1.0, runs a single conditioned pass.
        """
        if guidance_scale > 1.0 and g is not None:
            # Unconditional pass (zeroed speaker embedding)
            g_uncond = torch.zeros_like(g)
            v_uncond = self.estimator(x, t, content, f0, x_mask, g_uncond)
            # Conditioned pass
            v_cond = self.estimator(x, t, content, f0, x_mask, g)
            # CFG interpolation
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            return self.estimator(x, t, content, f0, x_mask, g)

    @torch.no_grad()
    def sample(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        n_steps: int = 16,
        method: str = "euler",
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate mel-spectrogram by solving the ODE.

        Args:
            content: (B, cond_ch, T) content features
            f0: (B, 1, T) f0
            x_mask: (B, 1, T)
            g: (B, gin_ch, 1) speaker embedding
            n_steps: number of ODE steps
            method: "euler" or "midpoint"
            guidance_scale: CFG scale. 1.0 = no guidance (standard),
                >1.0 = amplify speaker conditioning. Requires cfg_dropout > 0
                during training. Typical range: 1.0-3.0.
        """
        B = content.size(0)
        T = content.size(2)
        mel_ch = self.estimator.out_proj.out_channels
        device = content.device

        # Start from noise
        x = torch.randn(B, mel_ch, T, device=device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)

            if method == "euler":
                v = self._cfg_velocity(x, t, content, f0, x_mask, g, guidance_scale)
                x = x + v * dt
            elif method == "midpoint":
                # Half step
                v1 = self._cfg_velocity(x, t, content, f0, x_mask, g, guidance_scale)
                x_mid = x + v1 * (dt / 2)
                t_mid = torch.full((B,), (i + 0.5) * dt, device=device)
                v2 = self._cfg_velocity(x_mid, t_mid, content, f0, x_mask, g, guidance_scale)
                x = x + v2 * dt

            x = x * x_mask

        return x
