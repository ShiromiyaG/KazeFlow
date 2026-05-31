"""
Rectified Flow Matching (RFM) for voice conversion.

Replaces earlier architectures with a single generative model based on
Rectified Flow (Liu et al., 2022 — "Flow Straight and Fast").

Key differences vs CFM:
- Linear interpolation: x_t = (1-t)*x_0 + t*x_1  (no sigma_min shift)
- Target velocity: u_t = x_1 - x_0  (constant along the trajectory)
- Velocity-weighted MSE loss: normalises error by trajectory speed, giving
  balanced gradients across timesteps regardless of input distribution.
- Straighter ODE paths → fewer inference steps needed (4 steps ≈ CFM 16+)

Architecture: identical FlowEstimator (GatedDSConv + AdaGN + WaveNet skips)
reused from flow_matching.py — same capacity, different objective.

Inference: Euler/midpoint/RK4 ODE solver (same interface as CFM).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from kazeflow.models.commons import sequence_mask


# ---------------------------------------------------------------------------
# Timestep embedding (shared with flow_matching.py logic)
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
# Adaptive Group Norm
# ---------------------------------------------------------------------------

class AdaGN(nn.Module):
    """
    Adaptive Group Norm: GroupNorm + affine modulation from timestep/speaker.
    Fuses timestep and speaker conditioning into a single scale+shift.
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.proj = nn.Linear(cond_dim, channels * 2)
        self.proj.weight.data.zero_()
        self.proj.bias.data[:channels] = 1.0   # scale = 1
        self.proj.bias.data[channels:] = 0.0   # shift = 0

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        h = self.norm(x)
        scale_shift = self.proj(cond)           # (B, 2C)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return h * scale[:, :, None] + shift[:, :, None]


# ---------------------------------------------------------------------------
# Gated Depthwise Separable Conv Block
# ---------------------------------------------------------------------------

class GatedDSConvBlock(nn.Module):
    """
    Gated depthwise separable convolution block with AdaGN conditioning.

    AdaGN(cond) → DwConv(dilated) → PwConv(→2C) → tanh*sigmoid → PwConv(→C)
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
        h_a, h_b = h.chunk(2, dim=1)
        h = torch.tanh(h_a) * torch.sigmoid(h_b)
        h = self.drop(h)
        skip = self.skip_proj(h) * x_mask
        h = self.pwconv2(h)
        return (residual + h) * x_mask, skip


# ---------------------------------------------------------------------------
# Flow Estimator (velocity field v_θ)
# ---------------------------------------------------------------------------

class FlowEstimator(nn.Module):
    """
    Velocity field estimator for Rectified Flow Matching.

    Predicts v_θ(x_t, t, cond) where:
    - x_t = (1-t)*x_0 + t*x_1  (linear interpolation)
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

        # Timestep embedding: t → time_dim
        time_dim = hidden_channels * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_channels),
            nn.Linear(hidden_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition projection: content features + f0 → hidden
        # +1 for f0 (continuous voiced frequency)
        self.cond_proj = nn.Conv1d(cond_channels + 1, hidden_channels, 1)

        # Speaker projection → combined with timestep
        if gin_channels > 0:
            self.spk_proj = nn.Linear(gin_channels, time_dim)
        else:
            self.spk_proj = None

        # Input projection: noisy mel → hidden
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        # Merge input + condition
        self.merge = nn.Conv1d(hidden_channels * 2, hidden_channels, 1)

        # Gated DS conv blocks with exponential dilations
        cond_dim = time_dim
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** (i % 8)
            self.blocks.append(GatedDSConvBlock(
                hidden_channels, cond_dim, kernel_size, dilation, dropout,
            ))

        # Output projection: hidden → mel (velocity prediction)
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
        # Timestep + speaker conditioning → (B, time_dim)
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
# Rectified Flow Matching Module (training + inference)
# ---------------------------------------------------------------------------

class RectifiedFlowMatching(nn.Module):
    """
    Rectified Flow Matching for voice conversion.

    Training:
    - Sample t ~ logit_normal(mean, std), antithetic pairs
    - Linear interpolation: x_t = (1-t)*x_0 + t*x_1
    - Target velocity: u_t = x_1 - x_0  (constant, straight path)
    - Loss: velocity-weighted MSE || (v_θ - u_t) / ||u_t|| ||²
      (optional; normalises for input-amplitude invariance)

    Inference:
    - Solve ODE from x_0 ~ N(0,1) to x_1 using Euler/midpoint/RK4
    - 4 steps (default) gives excellent quality; 1–2 steps viable after reflow

    Drop-in replacement for ConditionalFlowMatching — identical
    .forward() and .sample() interfaces.
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
        # RFM-specific
        velocity_weighting: bool = True,
        t_sampling: str = "logit_normal",
        t_logit_mean: float = 0.0,
        t_logit_std: float = 1.0,
        cfg_dropout: float = 0.0,
        # Accepted for config compatibility (unused in RFM)
        sigma_min: float = 0.0,
    ):
        super().__init__()
        self.velocity_weighting = velocity_weighting
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

    def _sample_timesteps(self, B: int, device: torch.device,
                          dtype: torch.dtype) -> torch.Tensor:
        """
        Sample timesteps using antithetic logit-normal pairs.

        Each minibatch gets t and (1-t) pairs for variance reduction —
        every batch covers both ends of the ODE trajectory equally.
        """
        B_half = (B + 1) // 2
        if self.t_sampling == "logit_normal":
            u = torch.randn(B_half, device=device, dtype=dtype)
            t_half = torch.sigmoid(u * self.t_logit_std + self.t_logit_mean)
        else:
            t_half = torch.rand(B_half, device=device, dtype=dtype)
        t = torch.cat([t_half, 1.0 - t_half], dim=0)[:B]
        # Clamp away from endpoints (numerically unstable)
        return t.clamp(1e-4, 1.0 - 1e-4)

    def forward(
        self,
        x_1: torch.Tensor,         # (B, mel_ch, T) target mel
        x_mask: torch.Tensor,       # (B, 1, T)
        content: torch.Tensor,      # (B, cond_ch, T)
        f0: torch.Tensor,           # (B, 1, T)
        g: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Training forward: compute RFM loss."""
        B, C, T = x_1.shape

        t = self._sample_timesteps(B, x_1.device, x_1.dtype)
        t_expand = t[:, None, None]   # (B, 1, 1)

        # Source noise
        x_0 = torch.randn_like(x_1)

        # Linear interpolation — straight paths from noise to target
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1

        # Rectified velocity: constant along the linear path
        u_t = x_1 - x_0

        # Classifier-Free Guidance: randomly drop speaker embedding
        g_train = g
        if g is not None and self.cfg_dropout > 0.0 and self.training:
            drop_mask = torch.rand(B, device=x_1.device) < self.cfg_dropout
            if drop_mask.any():
                g_train = g.clone()
                g_train[drop_mask] = 0.0

        # Predict velocity
        v_t = self.estimator(x_t, t, content, f0, x_mask, g_train)

        # Velocity-weighted MSE loss.
        # Normalising by ||u_t|| makes the loss scale-invariant:
        # large-amplitude trajectories don't dominate small ones.
        # This is the key difference from standard CFM MSE.
        error = (v_t - u_t) * x_mask   # (B, C, T)
        if self.velocity_weighting:
            # Per-frame velocity norm across channels (+ eps for stability)
            vel_norm = u_t.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-4)
            error = error / vel_norm
        loss = error.pow(2).sum() / (x_mask.sum() * C)

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

        When guidance_scale > 1.0 runs two forward passes:
          v = v_uncond + guidance_scale * (v_cond - v_uncond)
        """
        if guidance_scale > 1.0 and g is not None:
            g_uncond = torch.zeros_like(g)
            v_uncond = self.estimator(x, t, content, f0, x_mask, g_uncond)
            v_cond = self.estimator(x, t, content, f0, x_mask, g)
            return v_uncond + guidance_scale * (v_cond - v_uncond)
        return self.estimator(x, t, content, f0, x_mask, g)

    @torch.no_grad()
    def sample(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        n_steps: int = 4,
        method: str = "euler",
        guidance_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate mel-spectrogram by solving the ODE.

        Args:
            content: (B, cond_ch, T) content features
            f0: (B, 1, T) F0 contour
            x_mask: (B, 1, T)
            g: (B, gin_ch, 1) speaker embedding
            n_steps: ODE steps (default 4; 1-2 viable after reflow distillation)
            method: "euler", "midpoint", or "rk4"
            guidance_scale: CFG scale. 1.0 = no guidance.
        """
        B = content.size(0)
        T = content.size(2)
        mel_ch = self.estimator.out_proj.out_channels
        device = content.device

        # Start from Gaussian noise
        x = torch.randn(B, mel_ch, T, device=device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)

            if method == "euler":
                v = self._cfg_velocity(x, t, content, f0, x_mask, g, guidance_scale)
                x = x + v * dt

            elif method == "midpoint":
                # RK2 midpoint — 2 NFE/step, 2nd-order accurate
                v1 = self._cfg_velocity(x, t, content, f0, x_mask, g, guidance_scale)
                x_mid = x + v1 * (dt / 2)
                t_mid = torch.full((B,), (i + 0.5) * dt, device=device)
                v2 = self._cfg_velocity(x_mid, t_mid, content, f0, x_mask, g, guidance_scale)
                x = x + v2 * dt

            elif method == "rk4":
                # Classical Runge-Kutta 4 — 4 NFE/step, 4th-order accurate.
                # For rectified flow (near-linear velocity), gains over midpoint
                # are modest, but useful for very-low-step (1–2) inference.
                t_half = torch.full((B,), (i + 0.5) * dt, device=device)
                t_end  = torch.full((B,), (i + 1.0) * dt, device=device)
                k1 = self._cfg_velocity(x,             t,      content, f0, x_mask, g, guidance_scale)
                k2 = self._cfg_velocity(x + k1*(dt/2), t_half, content, f0, x_mask, g, guidance_scale)
                k3 = self._cfg_velocity(x + k2*(dt/2), t_half, content, f0, x_mask, g, guidance_scale)
                k4 = self._cfg_velocity(x + k3*dt,     t_end,  content, f0, x_mask, g, guidance_scale)
                x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)

            else:
                raise ValueError(
                    f"Unknown ODE method '{method}'. Use 'euler', 'midpoint', or 'rk4'."
                )

            x = x * x_mask

        return x
