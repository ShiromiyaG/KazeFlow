"""
ChouwaGAN Discriminator for KazeFlow.

Combined MS-STFT + FastMPD + UnivHD discriminator.
- MS-STFT (3 scales): frequency-domain, phase-aware
- FastMPD (4 periods): periodic waveform patterns
- UnivHD (1): harmonic tracking
Total: 8 sub-discriminators, ~2.7M params.

Ported from Codename-fork-4 with all dependencies inlined.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.nn.utils import parametrize


def get_norm(use_spectral_norm: bool):
    return spectral_norm if use_spectral_norm else weight_norm


class _L2Normalize(nn.Module):
    """Parametrization that L2-normalizes a weight tensor to unit norm.

    Used by SAN (Slicing Adversarial Network) on the discriminator's final
    projection layer (conv_post).  This makes the projection a direction-only
    (cosine) similarity, bounding discriminator outputs without spectral norm
    or gradient penalties.
    """

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        flat = weight.reshape(weight.shape[0], -1)
        return F.normalize(flat, dim=1).reshape_as(weight)


# ══════════════════════════════════════════════════════════════════════════════
# Fast Multi-Period Discriminator (FastMPD)
# ══════════════════════════════════════════════════════════════════════════════

class DiscriminatorP(nn.Module):
    """Lightweight single-period sub-discriminator."""

    def __init__(self, period: int, channels: int = 32, max_channels: int = 128,
                 n_layers: int = 4, kernel_size: int = 5, stride: int = 3,
                 use_spectral_norm: bool = False, use_san: bool = False):
        super().__init__()
        self.period = period
        norm_f = get_norm(use_spectral_norm)

        self.convs = nn.ModuleList()
        in_ch = 1
        for i in range(n_layers):
            out_ch = min(channels * (2 ** i), max_channels)
            self.convs.append(norm_f(nn.Conv2d(
                in_ch, out_ch, (kernel_size, 1),
                stride=(stride, 1), padding=(kernel_size // 2, 0),
            )))
            in_ch = out_ch

        self.conv_final = norm_f(nn.Conv2d(
            in_ch, in_ch, (kernel_size, 1), stride=(1, 1),
            padding=(kernel_size // 2, 0),
        ))
        self.conv_post = nn.Conv2d(in_ch, 1, (3, 1), padding=(1, 0))
        if use_san:
            parametrize.register_parametrization(self.conv_post, "weight", _L2Normalize())
        else:
            self.conv_post = norm_f(self.conv_post)

    def forward(self, x: torch.Tensor,
                compute_fmaps: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            if compute_fmaps:
                fmap.append(x)

        x = F.leaky_relu(self.conv_final(x), 0.1)
        if compute_fmaps:
            fmap.append(x)

        x = self.conv_post(x)
        if compute_fmaps:
            fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


# ══════════════════════════════════════════════════════════════════════════════
# Universal Harmonic Discriminator (UnivHD)
# ══════════════════════════════════════════════════════════════════════════════

class HarmonicFilterbank(nn.Module):
    """Learnable triangular band-pass harmonic filterbank."""

    def __init__(self, n_harmonics: int = 10, bins_per_octave: int = 24,
                 f_min: float = 32.7, sample_rate: int = 48000, n_fft: int = 2048):
        super().__init__()
        self.n_harmonics = n_harmonics  

        harmonics = [0.5] + list(range(1, n_harmonics + 1))
        self.register_buffer("harmonics", torch.tensor(harmonics, dtype=torch.float32))
        self.n_total_harmonics = len(harmonics)

        f_max = sample_rate / (2.0 * n_harmonics)
        n_bins = int(bins_per_octave * math.log2(f_max / f_min))
        self.n_bins = n_bins

        k = torch.arange(n_bins, dtype=torch.float32)
        f_c = f_min * (2.0 ** (k / bins_per_octave))
        self.register_buffer("f_c", f_c)

        f_stft = torch.arange(n_fft // 2 + 1, dtype=torch.float32) * (sample_rate / n_fft)
        self.register_buffer("f_stft", f_stft)

        self.gamma = nn.Parameter(torch.ones(self.n_total_harmonics))
        self._cached_filt: Optional[torch.Tensor] = None

    def _build_filter(self) -> torch.Tensor:
        gamma = torch.clamp(self.gamma, min=1.0)
        centers = self.harmonics.unsqueeze(1) * self.f_c.unsqueeze(0)
        bw = gamma.unsqueeze(1) * 24.7 * (4.37 * centers / 1000.0 + 1.0)
        diff = torch.abs(
            self.f_stft.unsqueeze(0).unsqueeze(2) - centers.unsqueeze(1)
        )
        return torch.clamp(1.0 - diff / (bw.unsqueeze(1) + 1e-4), min=0.0)

    def forward(self, stft_mag: torch.Tensor) -> torch.Tensor:
        if self.training or self._cached_filt is None:
            filt = self._build_filter()
            if not self.training:
                self._cached_filt = filt
        else:
            filt = self._cached_filt

        # Use Python-int constants (compile-time known) instead of filt.shape,
        # and derive F from the actual input so torch.compile can track it
        # correctly when dynamo traces through _build_filter's symbolic shapes.
        H = self.n_total_harmonics   # int: len(harmonics) set in __init__
        K = self.n_bins               # int: n_bins set in __init__
        B, F, T = stft_mag.shape      # F = n_fft//2+1 from the real input tensor

        filt_t = filt.permute(0, 2, 1)
        filt_exp = filt_t.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * H, K, F)
        mag_exp = stft_mag.unsqueeze(1).expand(-1, H, -1, -1).reshape(B * H, F, T)
        out = torch.bmm(filt_exp, mag_exp)
        return out.reshape(B, H, K, T)


class HybridConvBlock(nn.Module):
    """HCB: DSConv (intra-harmonic) + Conv (inter-harmonic)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int] = (7, 7),
                 use_spectral_norm: bool = False):
        super().__init__()
        norm_f = get_norm(use_spectral_norm)
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.dsconv = norm_f(nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=pad, groups=in_channels,
        ))
        self.pconv = norm_f(nn.Conv2d(in_channels, out_channels, (1, 1)))
        self.conv = norm_f(nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ds_out = F.leaky_relu(self.dsconv(x), 0.1)
        ds_out = F.leaky_relu(self.pconv(ds_out), 0.1)
        conv_out = F.leaky_relu(self.conv(x), 0.1)
        return ds_out + conv_out


class MultiScaleDilatedConv(nn.Module):
    """MDC: 3 dilated convs + stride-2 downsampling."""

    def __init__(self, channels: int, kernel_size: Tuple[int, int] = (5, 5),
                 dilations: Optional[List[int]] = None,
                 use_spectral_norm: bool = False):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]
        norm_f = get_norm(use_spectral_norm)

        self.dilated_convs = nn.ModuleList()
        for d in dilations:
            pad = (d * (kernel_size[0] - 1) // 2, d * (kernel_size[1] - 1) // 2)
            self.dilated_convs.append(norm_f(nn.Conv2d(
                channels, channels, kernel_size, dilation=d, padding=pad,
            )))

        pad_ds = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv_ds = norm_f(nn.Conv2d(
            channels, channels, kernel_size, stride=(2, 1), padding=pad_ds,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for dc in self.dilated_convs:
            x = F.leaky_relu(dc(x), 0.1)
        x = x + residual
        x = F.leaky_relu(self.conv_ds(x), 0.1)
        return x


class UniversalHarmonicDiscriminator(nn.Module):
    """UnivHD: Waveform → STFT → HarmonicFilterbank → HCB → MDC×3 → Score."""

    def __init__(self, sr: int = 48000, n_harmonics: int = 10,
                 bins_per_octave: int = 24, channels: int = 32,
                 n_fft: int = 2048, hop_length: int = 512,
                 f_min: float = 32.7, n_mdc: int = 3,
                 use_spectral_norm: bool = False, use_san: bool = False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        self.filterbank = HarmonicFilterbank(
            n_harmonics=n_harmonics, bins_per_octave=bins_per_octave,
            f_min=f_min, sample_rate=sr, n_fft=n_fft,
        )
        n_total_harmonics = self.filterbank.n_total_harmonics

        self.hcb = HybridConvBlock(
            n_total_harmonics, channels, kernel_size=(7, 7),
            use_spectral_norm=use_spectral_norm,
        )

        self.mdc_blocks = nn.ModuleList([
            MultiScaleDilatedConv(channels, kernel_size=(5, 5),
                                  use_spectral_norm=use_spectral_norm)
            for _ in range(n_mdc)
        ])

        freq_dim = self.filterbank.n_bins
        mdc_k = 5
        for _ in range(n_mdc):
            pad = mdc_k // 2
            freq_dim = (freq_dim + 2 * pad - mdc_k) // 2 + 1

        norm_f = get_norm(use_spectral_norm)
        self.conv_post = nn.Conv2d(channels, 1, (freq_dim, 3), padding=(0, 1))
        if use_san:
            parametrize.register_parametrization(self.conv_post, "weight", _L2Normalize())
        else:
            self.conv_post = norm_f(self.conv_post)

    def forward_from_mag(self, stft_mag: torch.Tensor,
                         compute_fmaps: bool = True
                         ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h_tensor = self.filterbank(stft_mag)
        fmaps = []

        x = self.hcb(h_tensor)
        if compute_fmaps:
            fmaps.append(x)

        for mdc in self.mdc_blocks:
            x = mdc(x)
            if compute_fmaps:
                fmaps.append(x)

        x = self.conv_post(x)
        if compute_fmaps:
            fmaps.append(x)
        return torch.flatten(x, 1, -1), fmaps

    def _stft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = x.squeeze(1)
        if x_sq.dtype != torch.float32:
            x_sq = x_sq.float()
        pad = (self.n_fft - self.hop_length) // 2
        x_sq = F.pad(x_sq, (pad, pad), mode="constant")
        stft = torch.stft(
            x_sq, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=x_sq.dtype, device=x_sq.device),
            center=False, return_complex=True,
        )
        return stft.abs()


# ══════════════════════════════════════════════════════════════════════════════
# MS-STFT Sub-Discriminator
# ══════════════════════════════════════════════════════════════════════════════

class DiscriminatorSTFT(nn.Module):
    """Single-scale STFT discriminator (phase-aware, real+imag channels)."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256,
                 win_length: int = 1024, channels: int = 32,
                 max_channels: int = 128, n_layers: int = 3,
                 use_spectral_norm: bool = False, use_san: bool = False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

        norm_f = get_norm(use_spectral_norm)
        self.convs = nn.ModuleList()
        self.convs.append(norm_f(nn.Conv2d(2, channels, (3, 7), padding=(1, 3))))

        in_ch = channels
        for i in range(n_layers - 1):
            out_ch = min(channels * (2 ** (i + 1)), max_channels)
            self.convs.append(norm_f(nn.Conv2d(
                in_ch, out_ch, (3, 7), stride=(1, 2), padding=(1, 3),
            )))
            in_ch = out_ch

        self.convs.append(norm_f(nn.Conv2d(in_ch, in_ch, (3, 3), padding=(1, 1))))
        self.conv_post = nn.Conv2d(in_ch, 1, (3, 3), padding=(1, 1))
        if use_san:
            parametrize.register_parametrization(self.conv_post, "weight", _L2Normalize())
        else:
            self.conv_post = norm_f(self.conv_post)

    def forward(self, x: torch.Tensor,
                compute_fmaps: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self._stft(x)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            if compute_fmaps:
                fmap.append(x)
        x = self.conv_post(x)
        if compute_fmaps:
            fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

    def forward_from_stft(self, stft_complex: torch.Tensor,
                          compute_fmaps: bool = True
                          ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = torch.stack([stft_complex.real, stft_complex.imag], dim=1)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            if compute_fmaps:
                fmap.append(x)
        x = self.conv_post(x)
        if compute_fmaps:
            fmap.append(x)
        return torch.flatten(x, 1, -1), fmap

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = x.squeeze(1)
        if x_sq.dtype != torch.float32:
            x_sq = x_sq.float()
        pad = (self.n_fft - self.hop_length) // 2
        x_sq = F.pad(x_sq, (pad, pad), mode="constant")
        stft = torch.stft(
            x_sq, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=x_sq.dtype, device=x_sq.device),
            center=False, return_complex=True,
        )
        return torch.stack([stft.real, stft.imag], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Combined Discriminator
# ══════════════════════════════════════════════════════════════════════════════

class ChouwaGANDiscriminator(nn.Module):
    """
    Combined MS-STFT + FastMPD + UnivHD discriminator.
    Default: 2 STFT + 3 MPD + 1 UnivHD = 6 sub-discriminators.
    """

    def __init__(self, use_spectral_norm: bool = False, use_san: bool = False,
                 sample_rate: int = 48000, **cfg):
        super().__init__()

        stft_configs = cfg.get("stft_configs", [
            {"n_fft": 2048, "hop_length": 512, "win_length": 2048},
            {"n_fft": 1024, "hop_length": 256, "win_length": 1024},
        ])
        stft_channels = cfg.get("stft_channels", 32)
        stft_max_channels = cfg.get("stft_max_channels", 128)
        stft_n_layers = cfg.get("stft_n_layers", 3)

        mpd_periods = cfg.get("mpd_periods", [2, 3, 5])
        mpd_channels = cfg.get("mpd_channels", 32)
        mpd_max_channels = cfg.get("mpd_max_channels", 128)
        mpd_n_layers = cfg.get("mpd_n_layers", 4)

        self.discriminators = nn.ModuleList(
            [DiscriminatorSTFT(
                n_fft=sc["n_fft"], hop_length=sc["hop_length"],
                win_length=sc["win_length"], channels=stft_channels,
                max_channels=stft_max_channels, n_layers=stft_n_layers,
                use_spectral_norm=use_spectral_norm, use_san=use_san,
            ) for sc in stft_configs]
            + [DiscriminatorP(
                period=p, channels=mpd_channels, max_channels=mpd_max_channels,
                n_layers=mpd_n_layers, use_spectral_norm=use_spectral_norm,
                use_san=use_san,
            ) for p in mpd_periods]
        )

        # The shared STFT is computed once and reused by:
        #   (1) the first DiscriminatorSTFT whose config matches the shared key, and
        #   (2) UnivHD.forward_from_mag — so n_fft must match univhd_n_fft.
        _shared_n_fft = cfg.get("univhd_n_fft", 2048)
        _shared_hop   = cfg.get("univhd_hop_length", 512)

        self.univhd = UniversalHarmonicDiscriminator(
            sr=sample_rate,
            n_harmonics=cfg.get("univhd_n_harmonics", 10),
            bins_per_octave=cfg.get("univhd_bins_per_octave", 24),
            channels=cfg.get("univhd_channels", 32),
            n_fft=_shared_n_fft,
            hop_length=_shared_hop,
            use_spectral_norm=use_spectral_norm,
            use_san=use_san,
        )

        # Store as plain Python ints so _compute_stft can use them as constants
        # (torch.compile treats Python-int attributes as compile-time constants).
        self._shared_n_fft = _shared_n_fft
        self._shared_hop   = _shared_hop
        self._shared_stft_key = (_shared_n_fft, _shared_hop, _shared_n_fft)
        self.register_buffer("_shared_window", torch.hann_window(_shared_n_fft))

    def _compute_stft(self, x: torch.Tensor) -> torch.Tensor:
        x_sq = x.squeeze(1)
        if x_sq.dtype != torch.float32:
            x_sq = x_sq.float()
        n_fft      = self._shared_n_fft
        hop_length = self._shared_hop
        pad = (n_fft - hop_length) // 2
        x_sq = F.pad(x_sq, (pad, pad), mode="constant")
        window = self._shared_window.to(dtype=x_sq.dtype, device=x_sq.device)
        return torch.stft(
            x_sq, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
            window=window, center=False, return_complex=True,
        )

    def _is_shared_config(self, d: nn.Module) -> bool:
        if not isinstance(d, DiscriminatorSTFT):
            return False
        return (d.n_fft, d.hop_length, d.win_length) == self._shared_stft_key

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor,
                compute_fmaps: bool = True):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []

        stft_real = self._compute_stft(y)
        stft_fake = self._compute_stft(y_hat)

        for d in self.discriminators:
            if self._is_shared_config(d):
                y_d_r, fmap_r = d.forward_from_stft(stft_real, compute_fmaps=compute_fmaps)
                y_d_g, fmap_g = d.forward_from_stft(stft_fake, compute_fmaps=compute_fmaps)
            else:
                y_d_r, fmap_r = d(y, compute_fmaps=compute_fmaps)
                y_d_g, fmap_g = d(y_hat, compute_fmaps=compute_fmaps)

            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            if compute_fmaps:
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

        stft_mag_real = stft_real.abs()
        stft_mag_fake = stft_fake.abs()
        y_d_r_h, fmap_r_h = self.univhd.forward_from_mag(stft_mag_real, compute_fmaps=compute_fmaps)
        y_d_g_h, fmap_g_h = self.univhd.forward_from_mag(stft_mag_fake, compute_fmaps=compute_fmaps)
        y_d_rs.append(y_d_r_h)
        y_d_gs.append(y_d_g_h)
        if compute_fmaps:
            fmap_rs.append(fmap_r_h)
            fmap_gs.append(fmap_g_h)

        del stft_real, stft_fake, stft_mag_real, stft_mag_fake
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
