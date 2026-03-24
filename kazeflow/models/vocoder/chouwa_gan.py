"""
ChouwaGAN Vocoder for KazeFlow.

Self-contained HiFiGAN-PCPH vocoder with:
- SnakeBeta activations
- ConvTranspose1d upsampling (BigVGAN approach)
- LearnableHarmonicSource with anti-aliased harmonics
- Linear harmonic merge (no Tanh — prevents intermodulation/KL collapse)
- EMA (Exponential Moving Average) generator for improved inference quality
"""

import copy
import math
from itertools import chain
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from kazeflow.models.activations import snake_beta_forward


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(mean, std)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def remove_weight_norm_safe(module):
    if is_parametrized(module, "weight"):
        remove_parametrizations(module, "weight", leave_parametrized=True)
    else:
        remove_weight_norm(module)


# ---------------------------------------------------------------------------
# SnakeBeta Activation
# ---------------------------------------------------------------------------

class SnakeBeta(nn.Module):
    """
    Modified Snake activation: x + sin²(x·α) / (β + ε)
    Separate learnable params for frequency (alpha) and magnitude (beta).
    Computed in FP32 for numerical stability.
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        return snake_beta_forward(
            x, self.alpha, self.beta, self.alpha_logscale, self.no_div_by_zero
        )


# ---------------------------------------------------------------------------
# Anti-Aliasing Low-Pass Filter (BigVGAN style)
# ---------------------------------------------------------------------------

class LowPassFilter1d(nn.Module):
    """
    Kaiser-windowed sinc low-pass filter for anti-aliasing after nonlinearities.

    SnakeBeta computes x + sin^2(alpha*x) / beta, where sin^2 generates
    energy at 2*alpha times the input frequencies. When these exceed Nyquist,
    they alias back as metallic ringing artifacts. This filter removes those
    components before they propagate downstream.

    BigVGAN (Lee et al., 2023): "Periodic nonlinearities require anti-aliasing
    to prevent aliased harmonics from corrupting the signal."
    """

    def __init__(self, cutoff: float = 0.5, half_width: float = 0.6,
                 kernel_size: int = 12):
        super().__init__()
        # cutoff is relative to Nyquist (0.5 = Nyquist/2)
        even = kernel_size % 2 == 0

        # Kaiser beta for ~80dB stopband attenuation
        beta = 14.769656459379492

        n = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2

        # Sinc filter
        h = torch.sinc(2 * cutoff * n) * 2 * cutoff

        # Kaiser window
        window = torch.kaiser_window(kernel_size, periodic=False, beta=beta)
        h = h * window
        h = h / h.sum()  # normalize for unity DC gain

        self.register_buffer("filter", h.unsqueeze(0).unsqueeze(0))  # (1, 1, K)
        self.pad_size = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        x = F.pad(x, (self.pad_size, self.pad_size), mode="reflect")
        return F.conv1d(x, self.filter.expand(C, -1, -1), groups=C)


class SnakeBetaAA(nn.Module):
    """
    SnakeBeta + anti-aliasing low-pass filter.

    The LPF cutoff is adapted to the upsample rate of the stage: after
    upsampling by factor r, the valid spectral content occupies 1/r of
    the Nyquist range. The LPF cuts just above that boundary.
    """

    def __init__(self, channels: int, upsample_rate: int = 1):
        super().__init__()
        self.snake = SnakeBeta(channels, alpha_logscale=True)

        # Cutoff relative to Nyquist, adapted to upsample factor
        if upsample_rate > 1:
            cutoff = 0.5 / upsample_rate * 0.95  # 95% of Nyquist/rate
        else:
            cutoff = 0.45  # conservative for rate=1

        self.lpf = LowPassFilter1d(cutoff=cutoff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake(x)
        x = self.lpf(x)
        return x


# ---------------------------------------------------------------------------
# ResBlock with SnakeBeta
# ---------------------------------------------------------------------------

class ResBlock_SnakeBeta(nn.Module):
    """Residual block with SnakeBeta activations and dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: Tuple[int, ...] = (1, 3, 5),
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                  dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1,
                                  dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilations
        ])
        self.snake_acts1 = nn.ModuleList([
            SnakeBeta(channels, alpha_trainable=True, alpha_logscale=True)
            for _ in dilations
        ])
        self.snake_acts2 = nn.ModuleList([
            SnakeBeta(channels, alpha_trainable=True, alpha_logscale=True)
            for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, self.snake_acts1, self.snake_acts2
        ):
            residual = x
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(
                    self._block_forward, x, conv1, conv2, act1, act2,
                    use_reentrant=False,
                )
            else:
                x = self._block_forward(x, conv1, conv2, act1, act2)
            x = x + residual
        return x

    @staticmethod
    def _block_forward(x, conv1, conv2, act1, act2):
        x = act1(x)
        x = conv1(x)
        x = act2(x)
        x = conv2(x)
        return x

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm_safe(conv)


# ---------------------------------------------------------------------------
# UpsampleConvTranspose1d (BigVGAN approach)
# ---------------------------------------------------------------------------

class UpsampleConvTranspose1d(nn.Module):
    """
    Transposed-convolution upsampler (BigVGAN approach).

    ConvTranspose1d with stride == upsample_factor.  The kernel size is
    chosen so that ``padding = (kernel_size - stride) // 2``, which produces
    an output whose length is exactly ``input_length * stride``.

    Unlike sub-pixel convolution (PixelShuffle), each output position has
    independent weight paths through the transposed kernel, avoiding the
    piecewise-constant gradient problem that can cause mode collapse to silence.
    """

    def __init__(self, in_channels, out_channels, upsample_factor, kernel_size=None):
        super().__init__()
        stride = upsample_factor
        if kernel_size is None:
            kernel_size = upsample_factor * 2  # BigVGAN default: k = 2*stride

        # Ensure (kernel_size - stride) is even for symmetric padding
        if (kernel_size - stride) % 2 != 0:
            kernel_size += 1

        padding = (kernel_size - stride) // 2

        _conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        nn.init.kaiming_normal_(_conv.weight)
        if _conv.bias is not None:
            nn.init.zeros_(_conv.bias)
        self.conv = weight_norm(_conv)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# LearnableHarmonicSource (PCPH)
# ---------------------------------------------------------------------------

class LearnableHarmonicSource(nn.Module):
    """
    Band-limited harmonic source with learnable parameters.
    Output: [B, n_harmonics + 1, T_audio] (harmonics + noise channel).

    Features:
    - Per-harmonic amplitude weights (initialized to 1/k natural rolloff)
    - Per-harmonic phase offsets (small random init for natural partials)
    - Mel+f0-conditioned amplitude predictor for timbre-aware modulation
    - Learnable noise amplitude (voiced/unvoiced)
    - Smooth anti-aliasing mask (sigmoid) to avoid click artifacts
    """

    def __init__(self, sample_rate: int, hop_length: int = 480,
                 n_harmonics: int = 6, add_noise_std: float = 0.003,
                 mel_channels: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_harmonics = n_harmonics

        k = torch.arange(1, n_harmonics + 1, dtype=torch.float32).view(1, -1, 1)
        self.register_buffer("k", k)

        # Learnable per-harmonic amplitude (log-scale, init to 1/k rolloff)
        init_log_amp = -torch.log(
            torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        self.amp_logscale = nn.Parameter(init_log_amp)

        # Per-harmonic phase offset: small random init for natural partials
        self.phase_offset = nn.Parameter(torch.randn(n_harmonics) * 0.1)

        # Mel+f0-conditioned amplitude predictor
        self.amp_predictor = nn.Sequential(
            nn.Conv1d(1 + mel_channels, 32, kernel_size=5, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, n_harmonics, kernel_size=1),
        )
        nn.init.zeros_(self.amp_predictor[-1].weight)
        nn.init.zeros_(self.amp_predictor[-1].bias)

        # Learnable noise amplitude (log-scale)
        self.log_noise_voiced = nn.Parameter(torch.tensor(math.log(add_noise_std)))
        self.log_noise_unvoiced = nn.Parameter(torch.tensor(math.log(0.003)))

    def forward(self, f0: torch.Tensor, upsample_factor: Optional[int] = None,
                mel: Optional[torch.Tensor] = None):
        hop = upsample_factor if upsample_factor is not None else self.hop_length
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)

        device_type = f0.device.type
        with torch.amp.autocast(device_type, enabled=False):
            f0 = f0.float()

            # Amplitude modulation from f0 + mel context
            if mel is not None:
                mel_f = mel.float()
                if mel_f.shape[-1] != f0.shape[-1]:
                    mel_f = F.interpolate(mel_f, size=f0.shape[-1],
                                          mode="linear", align_corners=False)
                amp_mod = self.amp_predictor(torch.cat([f0, mel_f], dim=1))
            else:
                amp_mod = torch.zeros(
                    f0.shape[0], self.n_harmonics, f0.shape[-1],
                    device=f0.device, dtype=f0.dtype,
                )

            f0_up = F.interpolate(f0, scale_factor=float(hop),
                                  mode="linear", align_corners=False)

            phase_increment = f0_up / self.sample_rate
            raw_phase = self._wrapped_cumsum(phase_increment)
            base_phase = raw_phase * (2.0 * math.pi)

            offset = self.phase_offset.view(1, -1, 1)
            harmonics = torch.sin(self.k * base_phase + offset * (2.0 * math.pi))

            amp = torch.exp(self.amp_logscale).view(1, -1, 1)
            harmonics = harmonics * amp

            # Upsample modulation to audio rate and apply
            amp_mod_up = F.interpolate(
                amp_mod, size=harmonics.shape[-1],
                mode="linear", align_corners=False,
            )
            harmonics = harmonics * (2.0 * torch.sigmoid(amp_mod_up))

            # Smooth anti-aliasing: sigmoid transition instead of hard 0/1 cutoff
            nyquist = self.sample_rate / 2.0
            voiced_mask = (f0_up > 1.0).float()
            aa_margin = nyquist * 0.05
            aa_mask = torch.sigmoid((nyquist - self.k * f0_up) / aa_margin)
            harmonics = harmonics * (aa_mask * voiced_mask)

            # Learnable noise amplitude
            noise = torch.randn_like(f0_up)
            noise_voiced = torch.exp(self.log_noise_voiced)
            noise_unvoiced = torch.exp(self.log_noise_unvoiced)
            noise_amp = voiced_mask * noise_voiced + (1.0 - voiced_mask) * noise_unvoiced
            noise = noise * noise_amp

            return torch.cat([harmonics, noise], dim=1)

    @staticmethod
    def _wrapped_cumsum(x, chunk_size=256):
        T = x.shape[-1]
        if T <= chunk_size:
            return torch.fmod(torch.cumsum(x, dim=-1), 1.0)
        chunks = x.split(chunk_size, dim=-1)
        results = []
        carry = torch.zeros_like(x[..., :1])
        for chunk in chunks:
            chunk_cum = torch.cumsum(chunk, dim=-1) + carry
            carry = chunk_cum[..., -1:]
            chunk_cum = torch.fmod(chunk_cum, 1.0)
            carry = torch.fmod(carry, 1.0)
            results.append(chunk_cum)
        return torch.cat(results, dim=-1)


# ---------------------------------------------------------------------------
# HiFiGAN-PCPH Head
# ---------------------------------------------------------------------------

class HiFiGANPCPHHead(nn.Module):
    """
    HiFiGAN generator head with PCPH harmonic matrix injection,
    anti-aliased SnakeBeta activations, and multi-scale conditioning.

    Architecture:
    - SnakeBetaAA: anti-aliased activations prevent Nyquist aliasing artifacts
    - Post-upsample LPF: filters spectral images from transposed convolution
    - FiLM speaker conditioning: re-injects speaker identity at each stage
    - Mel skip connections: multi-scale spectral information injection
    - Soft clipping output: preserves dynamic range beyond [-1, 1]
    - ConvTranspose1d upsampling (BigVGAN approach)
    - Activation checkpointing (optional) for memory-efficient training
    """

    def __init__(
        self,
        input_channels: int = 512,
        upsample_rates: list = [8, 5, 4, 2],
        upsample_kernel_sizes: list = [16, 10, 8, 4],
        resblock_kernel_sizes: list = [3, 7, 11],
        resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_initial_channel: int = 384,
        gin_channels: int = 256,
        sr: int = 48000,
        pre_conv_kernel_size: int = 7,
        n_harmonics: int = 6,
        use_checkpoint: bool = False,
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int = 2048,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)
        self.use_checkpoint = use_checkpoint

        self.conv_pre = weight_norm(nn.Conv1d(
            input_channels, upsample_initial_channel,
            pre_conv_kernel_size, 1,
            padding=get_padding(pre_conv_kernel_size),
        ))

        self.pre_snake = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.har_convs = nn.ModuleList()
        self.post_upsample_lpf = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.mel_skips = nn.ModuleList()

        # Build explicit in/out channel lists with min_ch=32 floor.
        min_ch = 32
        self._in_channels = [upsample_initial_channel]
        self.channels = []
        for i in range(len(upsample_rates)):
            out_ch = max(upsample_initial_channel // (2 ** (i + 1)), min_ch)
            self.channels.append(out_ch)
            if i + 1 < len(upsample_rates):
                self._in_channels.append(out_ch)

        self.stride_f0s = [
            math.prod(upsample_rates[i + 1:]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = self._in_channels[i]
            out_ch = self.channels[i]

            # Anti-aliased SnakeBeta: filters Snake harmonics before upsample
            self.pre_snake.append(SnakeBetaAA(in_ch, upsample_rate=u))

            self.ups.append(
                UpsampleConvTranspose1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    upsample_factor=u,
                    kernel_size=k,
                )
            )

            # Post-upsample LPF: filters spectral images from transposed conv
            post_cutoff = 0.5 / u * 0.95
            self.post_upsample_lpf.append(LowPassFilter1d(cutoff=post_cutoff))

            # FiLM: per-stage speaker conditioning (scale + shift)
            if gin_channels != 0:
                film = nn.Conv1d(gin_channels, out_ch * 2, 1)
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
                self.film_layers.append(film)
            else:
                self.film_layers.append(None)

            # Mel skip: inject multi-scale spectral info from conv_pre output.
            # Initialized to zero so skip starts neutral — activates gradually.
            mel_skip = nn.Conv1d(upsample_initial_channel, out_ch, 1)
            nn.init.zeros_(mel_skip.weight)
            nn.init.zeros_(mel_skip.bias)
            self.mel_skips.append(mel_skip)

            # Harmonic injection conv
            stride = self.stride_f0s[i]
            if stride > 1:
                kernel = stride * 2 - stride % 2
                pad = (kernel - stride) // 2
                self.har_convs.append(
                    weight_norm(nn.Conv1d(
                        n_harmonics + 1, out_ch,
                        kernel_size=kernel, stride=stride, padding=pad,
                    ))
                )
            else:
                self.har_convs.append(
                    weight_norm(nn.Conv1d(
                        n_harmonics + 1, out_ch, kernel_size=1,
                    ))
                )

        self.resblocks = nn.ModuleList([
            ResBlock_SnakeBeta(self.channels[i], k, d, use_checkpoint=use_checkpoint)
            for i in range(len(self.ups))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        ])

        # Per-stage learnable gate for mel skips — starts at 0, grows freely
        self.mel_skip_gates = nn.ParameterList([
            nn.Parameter(torch.zeros(1))
            for _ in range(len(upsample_rates))
        ])

        # Learned resblock mixing weights (softmax-normalized per upsample stage)
        self.resblock_weights = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_kernels))
            for _ in range(self.num_upsamples)
        ])

        # Output head: SnakeBeta → Conv1d → soft clipping
        self.post_snake = SnakeBeta(self.channels[-1], alpha_logscale=True)
        self.conv_post = weight_norm(nn.Conv1d(
            self.channels[-1], 1, kernel_size=7, padding=3,
        ))
        nn.init.zeros_(self.conv_post.weight)
        nn.init.zeros_(self.conv_post.bias)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
            nn.init.normal_(self.cond.weight, mean=0.0, std=0.01)
            if self.cond.bias is not None:
                nn.init.constant_(self.cond.bias, 0.0)

    def _resblock_forward(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """Forward through one stage's resblocks with learned mixing."""
        start_idx = stage_idx * self.num_kernels
        w = F.softmax(self.resblock_weights[stage_idx], dim=0)
        return sum(
            w[j] * self.resblocks[start_idx + j](x)
            for j in range(self.num_kernels)
        )

    def forward(self, x, har_source, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        mel_feat = x  # save for skip connections

        for i in range(self.num_upsamples):
            x = self.pre_snake[i](x)
            x = self.ups[i](x)
            x = self.post_upsample_lpf[i](x)

            # FiLM speaker conditioning (re-inject at every stage).
            # Clamp scale to [-1, 1] to prevent multiplicative gradient explosion.
            if g is not None and self.film_layers[i] is not None:
                film = self.film_layers[i](g)  # (B, ch*2, 1) broadcasts
                scale, shift = film.chunk(2, dim=1)
                scale = scale.clamp(-1.0, 1.0)
                x = x * (1.0 + scale) + shift

            # Harmonic injection
            har_out = self.har_convs[i](har_source)
            min_len = min(x.shape[-1], har_out.shape[-1])
            x = x[..., :min_len] + har_out[..., :min_len]

            # Mel skip connection — gated so initial contribution starts at zero.
            # Clamp gate to [-2, 2] before tanh to prevent runaway gradients
            # at large gate values (tanh saturates → vanishing grad for gate
            # but large grad for mel_proj via the scale path).
            mel_proj = self.mel_skips[i](mel_feat)
            mel_proj = F.interpolate(
                mel_proj, size=x.shape[-1],
                mode="linear", align_corners=False,
            )
            gate = torch.tanh(self.mel_skip_gates[i].clamp(-2.0, 2.0))
            x = x + gate * mel_proj

            # Resblock mixing with activation checkpointing
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(
                    self._resblock_forward, x, i,
                    use_reentrant=False,
                )
            else:
                x = self._resblock_forward(x, i)

        # Output head: SnakeBeta → Conv1d → soft clipping
        x = self.post_snake(x)
        x = self.conv_post(x)
        x = x / (1.0 + x.abs())
        return x.unsqueeze(1) if x.dim() == 2 else x

    def remove_weight_norm(self):
        remove_weight_norm_safe(self.conv_pre)
        remove_weight_norm_safe(self.conv_post)
        for l in self.ups:
            remove_weight_norm_safe(l.conv)
        for l in self.har_convs:
            remove_weight_norm_safe(l)
        for l in self.resblocks:
            l.remove_weight_norm()


# ---------------------------------------------------------------------------
# ChouwaGAN Generator (Full)
# ---------------------------------------------------------------------------

class ChouwaGANGenerator(nn.Module):
    """
    ChouwaGAN: HiFiGAN with SnakeBeta + ConvTranspose1d + PCPH harmonics.

    For KazeFlow: No backbone (ConvNeXt) — the flow matching model replaces
    the VITS encoder/flow, so the vocoder receives mel-spectrograms directly.

    Interface: forward(mel, f0, g=None) → waveform
    """

    def __init__(
        self,
        mel_channels: int = 128,
        upsample_rates: list = [8, 5, 6, 2],
        upsample_initial_channel: int = 256,
        upsample_kernel_sizes: list = [16, 10, 12, 4],
        resblock_kernel_sizes: list = [3, 7],
        resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5]],
        gin_channels: int = 256,
        sr: int = 48000,
        n_harmonics: int = 6,
        use_checkpoint: bool = False,
        n_fft: int = 2048,
        win_length: int = 2048,
    ):
        super().__init__()
        self.upp = math.prod(upsample_rates)
        self.n_harmonics = n_harmonics

        # Band-limited harmonic source
        self.m_source = LearnableHarmonicSource(
            sample_rate=sr,
            hop_length=self.upp,
            n_harmonics=n_harmonics,
            mel_channels=mel_channels,
        )

        # HiFiGAN-PCPH head — receives FULL per-harmonic source (no early merge)
        # iSTFT hop_length must equal the total upsample product
        self.head = HiFiGANPCPHHead(
            input_channels=mel_channels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_initial_channel=upsample_initial_channel,
            gin_channels=gin_channels,
            sr=sr,
            n_harmonics=n_harmonics,
            use_checkpoint=use_checkpoint,
            n_fft=n_fft,
            hop_length=self.upp,
            win_length=win_length,
        )

    def forward(self, mel: torch.Tensor, f0: torch.Tensor,
                g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mel: (B, mel_ch, T) mel-spectrogram
            f0:  (B, T) or (B, 1, T) fundamental frequency
            g:   (B, gin_ch, 1) speaker embedding
        Returns:
            waveform: (B, 1, T_audio)
        """
        f0_len = f0.shape[-1]
        if mel.shape[-1] != f0_len:
            raise ValueError(
                f"Temporal mismatch: mel {mel.shape[-1]} vs f0 {f0_len}"
            )

        har_source = self.m_source(f0, self.upp, mel=mel)
        return self.head(mel, har_source, g=g)

    def remove_weight_norm(self):
        self.head.remove_weight_norm()

    def register_output_grad_clip(self, max_norm: float) -> None:
        """Register per-parameter gradient clip hooks on the output layer.

        The final conv layer (conv_post) receives ~97% of adversarial gradient
        norm. Per-layer clipping prevents it from dominating the global norm
        and starving deeper layers.
        """
        def _clip_hook(grad, _max=max_norm):
            gn = grad.norm()
            return grad * (_max / gn) if gn > _max else grad
        for p in self.head.conv_post.parameters():
            p.register_hook(_clip_hook)

    def get_compilable_module(self) -> nn.Module:
        """Return the main compute module for torch.compile."""
        return self.head


# ---------------------------------------------------------------------------
# EMA Generator (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMAGenerator:
    """
    Exponential Moving Average of vocoder generator weights.

    Maintains a shadow copy of the generator parameters, updated after each
    optimizer step with: ema_param = decay * ema_param + (1 - decay) * param

    The EMA model typically produces higher-quality audio than the raw trained
    model because it smooths out SGD noise. Used for inference only.

    Polyak & Juditsky (1992); widely adopted in GANs (StyleGAN, BigGAN).

    Usage:
        ema = EMAGenerator(vocoder, decay=0.999)
        # After each optimizer step:
        ema.update()
        # For inference:
        ema_vocoder = ema.get_model()  # returns the shadow model
        # Checkpoint:
        state = ema.state_dict()
        ema.load_state_dict(state)

    Note: This is NOT an nn.Module — it's a wrapper that holds a deepcopy
    of the generator. This avoids complications with DDP, torch.compile,
    and weight_norm parametrizations.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.shadow.requires_grad_(False)
        self._source = model

    @torch.no_grad()
    def update(self):
        """Update EMA parameters. Call after each optimizer step."""
        for ema_p, src_p in zip(self.shadow.parameters(), self._source.parameters()):
            ema_p.data.lerp_(src_p.data.float(), 1.0 - self.decay)

    def get_model(self) -> nn.Module:
        """Get the EMA shadow model (for inference)."""
        return self.shadow

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.shadow.load_state_dict(state_dict)

    def to(self, device):
        """Move EMA shadow to device."""
        self.shadow = self.shadow.to(device)
        return self
