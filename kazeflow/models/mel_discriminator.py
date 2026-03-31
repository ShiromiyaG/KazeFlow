"""
Multi-Scale Mel Discriminator for Direct Mel Regression.

Operates on mel spectrograms (2D) at multiple resolutions to force the
mel predictor to produce sharp, realistic spectrograms.  Without this,
pure L1 regression tends to produce blurry / over-smoothed mels.

Interface matches ChouwaGANDiscriminator:
    forward(y, y_hat, compute_fmaps=True)
      → (y_d_rs, y_d_gs, fmap_rs, fmap_gs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class MelSubDiscriminator(nn.Module):
    """Single-scale 2D discriminator for mel spectrograms.

    Treats the mel as a single-channel image [B, 1, n_mels, T] and
    applies a stack of strided Conv2d layers followed by a scoring head.
    """

    def __init__(
        self,
        channels: int = 64,
        max_channels: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        in_c = 1  # mel as single-channel
        for i in range(n_layers):
            out_c = min(channels * (2 ** i), max_channels)
            # Stride 2 on first n_layers-1 layers, stride 1 on last
            s = (2, 2) if i < n_layers - 1 else (1, 1)
            self.convs.append(
                nn.utils.weight_norm(
                    nn.Conv2d(in_c, out_c, kernel_size=3, stride=s, padding=1)
                )
            )
            in_c = out_c
        self.conv_post = nn.utils.weight_norm(
            nn.Conv2d(in_c, 1, kernel_size=3, stride=1, padding=1)
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 1, n_mels, T)
        Returns:
            score: (B, N) flattened discriminator scores
            fmaps: list of intermediate feature maps
        """
        fmaps = []
        for conv in self.convs:
            x = self.act(conv(x))
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        # Flatten in fp32 to avoid overflow when scores are summed externally
        return x.float().flatten(1, -1), fmaps


class MultiScaleMelDiscriminator(nn.Module):
    """
    Multi-scale mel spectrogram discriminator.

    Applies sub-discriminators at progressively downsampled versions
    of the mel spectrogram.  Forces the predictor to be realistic at
    all temporal/frequency resolutions.

    Interface:
        forward(y, y_hat, compute_fmaps=True)
        → (y_d_rs, y_d_gs, fmap_rs, fmap_gs)
    """

    def __init__(
        self,
        n_scales: int = 3,
        channels: int = 64,
        max_channels: int = 256,
        n_layers: int = 4,
        # Unused — accepted for forward-compat with registry pattern
        **_kwargs,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MelSubDiscriminator(channels, max_channels, n_layers)
            for _ in range(n_scales)
        ])
        # Average-pool between scales (4×4 kernel, stride 2, pad 1)
        self.downsample = nn.AvgPool2d(
            kernel_size=4, stride=2, padding=1, count_include_pad=False,
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        compute_fmaps: bool = True,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor],
        List[List[torch.Tensor]], List[List[torch.Tensor]],
    ]:
        """
        Args:
            y:     (B, n_mels, T) — real (ground truth) mel
            y_hat: (B, n_mels, T) — predicted mel
            compute_fmaps: if True, return intermediate feature maps
                           (needed for feature matching loss)

        Returns:
            y_d_rs:  list of score tensors for real
            y_d_gs:  list of score tensors for fake
            fmap_rs: list of fmap lists for real (empty lists if not computed)
            fmap_gs: list of fmap lists for fake (empty lists if not computed)
        """
        y_d_rs: List[torch.Tensor] = []
        y_d_gs: List[torch.Tensor] = []
        fmap_rs: List[List[torch.Tensor]] = []
        fmap_gs: List[List[torch.Tensor]] = []

        y_r = y.unsqueeze(1)      # (B, 1, n_mels, T)
        y_g = y_hat.unsqueeze(1)   # (B, 1, n_mels, T)

        for disc in self.discriminators:
            score_r, fmap_r = disc(y_r)
            score_g, fmap_g = disc(y_g)
            y_d_rs.append(score_r)
            y_d_gs.append(score_g)
            if compute_fmaps:
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)
            else:
                fmap_rs.append([])
                fmap_gs.append([])
            # Downsample for next scale
            y_r = self.downsample(y_r)
            y_g = self.downsample(y_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
