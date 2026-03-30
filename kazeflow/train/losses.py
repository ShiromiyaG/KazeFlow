"""
Loss functions for KazeFlow training.

Combines:
- CFM velocity loss (MSE on predicted vs target velocity)
- LSGAN discriminator/generator losses
- Feature matching loss
- Mel-spectrogram reconstruction loss
- Phase continuity loss (APNet-style instantaneous frequency + group delay)
- Gradient Balancer (EnCodec-style automatic gradient normalization)

All epsilon values use 1e-4 minimum for FP16 safety (FP16 min subnormal ≈ 6e-8,
but practical safe floor for divisions/logs under mixed precision is ~1e-4).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# FP16-safe epsilon: large enough to survive FP16 rounding and prevent
# division-by-zero / log-of-zero in mixed-precision training.
_EPS = 1e-4


@torch.amp.autocast("cuda", enabled=False)
def mel_spectrogram_loss(y: torch.Tensor, y_hat: torch.Tensor,
                         n_fft: int = 2048, hop_length: int = 480,
                         win_length: Optional[int] = None, n_mels: int = 128,
                         sample_rate: int = 48000,
                         mel_basis: Optional[torch.Tensor] = None,
                         ) -> torch.Tensor:
    """L1 mel-spectrogram reconstruction loss."""
    if win_length is None:
        win_length = n_fft
    mel_real = _mel_spec(y, n_fft, hop_length, win_length, n_mels,
                         sample_rate, mel_basis)
    mel_fake = _mel_spec(y_hat, n_fft, hop_length, win_length, n_mels,
                         sample_rate, mel_basis)
    return F.l1_loss(mel_fake, mel_real)


def _mel_spec(x: torch.Tensor, n_fft: int, hop_length: int,
              win_length: int, n_mels: int, sample_rate: int,
              mel_basis: Optional[torch.Tensor]) -> torch.Tensor:
    """Compute log mel-spectrogram."""
    if x.dim() == 3:
        x = x.squeeze(1)
    x = x.float()

    window = torch.hann_window(win_length, device=x.device)
    pad = (n_fft - hop_length) // 2
    x = F.pad(x, (pad, pad), mode="reflect")
    spec = torch.stft(
        x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=False, return_complex=True,
    )
    mag = spec.abs()

    if mel_basis is None:
        # Build mel filterbank on the fly (cached by caller in practice)
        import torchaudio
        mel_basis = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=sample_rate / 2.0,
            n_mels=n_mels, sample_rate=sample_rate,
        ).T.to(x.device)  # (n_mels, n_fft//2+1)

    mel = torch.matmul(mel_basis, mag)
    log_mel = torch.log(torch.clamp(mel, min=_EPS))
    return log_mel


@torch.amp.autocast("cuda", enabled=False)
def feature_loss(fmap_r: List[List[torch.Tensor]],
                 fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """Feature matching loss (L1 between discriminator feature maps).

    Forced to FP32 — feature maps under autocast are FP16, and accumulated
    L1 differences lose precision when many layers are summed.
    """
    loss = 0.0
    n_layers = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.float() - gl.float()))
            n_layers += 1
    if n_layers > 0:
        return 2 * loss / n_layers
    return torch.tensor(0.0)


# ---------------------------------------------------------------------------
# LSGAN losses
# ---------------------------------------------------------------------------

@torch.amp.autocast("cuda", enabled=False)
def discriminator_loss_lsgan(disc_real_outputs: List[torch.Tensor],
                       disc_generated_outputs: List[torch.Tensor],
                       ) -> Tuple[torch.Tensor, float, float]:
    """LSGAN discriminator loss with monitoring metrics.

    Forced to FP32 — the squared terms (1-d)^2 can overflow FP16 when
    discriminator outputs become large during adversarial training.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    d_real_sum, d_fake_sum = 0.0, 0.0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr_f = dr.float().clamp(-50.0, 50.0)
        dg_f = dg.float().clamp(-50.0, 50.0)
        r_loss = torch.mean((1.0 - dr_f) ** 2)
        g_loss = torch.mean(dg_f ** 2)
        loss = loss + r_loss + g_loss
        d_real_sum += dr.detach().mean().item()
        d_fake_sum += dg.detach().mean().item()

    n = max(len(disc_real_outputs), 1)
    return loss / n, d_real_sum / n, d_fake_sum / n


@torch.amp.autocast("cuda", enabled=False)
def generator_loss_lsgan(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
    """LSGAN generator loss.

    Forced to FP32 — same overflow concern as discriminator_loss_lsgan.
    """
    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for dg in disc_outputs:
        dg_f = dg.float().clamp(-50.0, 50.0)
        loss = loss + torch.mean((1.0 - dg_f) ** 2)
    return loss / max(len(disc_outputs), 1)


# ---------------------------------------------------------------------------
# Hinge GAN losses — bounded gradients, no squaring → naturally stable
# ---------------------------------------------------------------------------

@torch.amp.autocast("cuda", enabled=False)
def discriminator_loss_hinge(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, float, float]:
    """Hinge discriminator loss with monitoring metrics.

    D_loss = E[ReLU(1 - D(x))] + E[ReLU(1 + D(G(z)))]
    Gradients are zero once the margin is satisfied → no unbounded growth.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    d_real_sum, d_fake_sum = 0.0, 0.0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr_f = dr.float()
        dg_f = dg.float()
        loss = loss + torch.mean(F.relu(1.0 - dr_f)) + torch.mean(F.relu(1.0 + dg_f))
        d_real_sum += dr.detach().mean().item()
        d_fake_sum += dg.detach().mean().item()

    n = max(len(disc_real_outputs), 1)
    return loss / n, d_real_sum / n, d_fake_sum / n


@torch.amp.autocast("cuda", enabled=False)
def generator_loss_hinge(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Hinge generator loss.

    G_loss = -E[D(G(z))]
    Linear in D output → bounded gradients, no squaring.
    """
    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for dg in disc_outputs:
        loss = loss - torch.mean(dg.float())
    return loss / max(len(disc_outputs), 1)


@torch.amp.autocast("cuda", enabled=False)
def generator_loss_soft_hinge(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
    """Soft-hinge (logistic non-saturating) generator loss.

    G_loss = E[softplus(-D(G(z)))] = E[log(1 + exp(-D(G(z))))]

    When D(G(z)) >> 0 (generator winning): gradient → 0 (saturates).
    When D(G(z)) << 0 (generator losing):  gradient → 1 (bounded).
    Combines bounded gradients with natural saturation — StyleGAN2 style.
    Paired with standard hinge discriminator for margin-based stability.
    """
    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for dg in disc_outputs:
        loss = loss + torch.mean(F.softplus(-dg.float()))
    return loss / max(len(disc_outputs), 1)


@torch.amp.autocast("cuda", enabled=False)
def discriminator_loss_softplus(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, float, float]:
    """Softplus (logistic) discriminator loss — margin-free, scale-invariant.

    D_loss = E[softplus(-D(x))] + E[softplus(D(G(z)))]

    Unlike hinge (margin=1.0), softplus has no hard threshold — gradient
    decays smoothly as the discriminator becomes more confident.  This
    makes it compatible with SAN's L2-normalized conv_post, which caps
    output magnitude at ~±0.5 and can never reach the hinge margin.

    StyleGAN2 uses this for both D and G.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    d_real_sum, d_fake_sum = 0.0, 0.0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr_f = dr.float()
        dg_f = dg.float()
        loss = loss + torch.mean(F.softplus(-dr_f)) + torch.mean(F.softplus(dg_f))
        d_real_sum += dr.detach().mean().item()
        d_fake_sum += dg.detach().mean().item()

    n = max(len(disc_real_outputs), 1)
    return loss / n, d_real_sum / n, d_fake_sum / n


class LeCamEMA:
    """LeCam regularization for discriminator (Tseng et al., 2021).

    Tracks EMA of real/fake discriminator scores and penalizes the
    discriminator when it becomes overconfident relative to the running
    averages.  This prevents the steep loss surface that causes gradient
    explosions — attacking the *cause* rather than the symptom.

    Cost: ~zero (scalar EMA updates + one mean per forward).
    """

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.ema_real = 0.0
        self.ema_fake = 0.0
        self.initialized = False

    @torch.no_grad()
    def update(self, real_preds: List[torch.Tensor],
               fake_preds: List[torch.Tensor]):
        r = sum(p.mean().item() for p in real_preds) / max(len(real_preds), 1)
        f = sum(p.mean().item() for p in fake_preds) / max(len(fake_preds), 1)
        if not self.initialized:
            self.ema_real = r
            self.ema_fake = f
            self.initialized = True
        else:
            self.ema_real = self.decay * self.ema_real + (1 - self.decay) * r
            self.ema_fake = self.decay * self.ema_fake + (1 - self.decay) * f

    def penalty(self, real_preds: List[torch.Tensor],
                fake_preds: List[torch.Tensor]) -> torch.Tensor:
        """Compute LeCam penalty across all sub-discriminators (mean).

        Forced to FP32 — disc predictions may arrive in FP16 and
        .pow(2) can overflow FP16 max (65504) if outputs grow.
        """
        loss = torch.tensor(0.0, device=real_preds[0].device)
        for r_pred in real_preds:
            loss = loss + torch.relu(r_pred.float() - self.ema_fake).pow(2).mean()
        for f_pred in fake_preds:
            loss = loss + torch.relu(self.ema_real - f_pred.float()).pow(2).mean()
        n = max(len(real_preds) + len(fake_preds), 1)
        return loss / n


@torch.amp.autocast("cuda", enabled=False)
def r1_gradient_penalty(
    disc_real_outputs: List[torch.Tensor],
    real_audio: torch.Tensor,
) -> torch.Tensor:
    """R1 gradient penalty (Mescheder et al., 2018).

    Penalizes ||∇_x D(x)||² on real samples, forcing the discriminator
    to have bounded gradients near the data manifold.  This prevents the
    sharp decision boundaries that cause generator gradient explosions.

    ``real_audio`` must have ``requires_grad=True`` **before** the
    discriminator forward pass so the computation graph is retained.

    Forced to FP32 — second-order gradients can overflow in reduced precision.
    """
    # Sum all discriminator outputs (spatial + disc index)
    r1_target = sum(d.float().sum() for d in disc_real_outputs)

    grad_real, = torch.autograd.grad(
        outputs=r1_target,
        inputs=real_audio,
        create_graph=True,
    )

    # ||grad||² per sample, mean over batch
    penalty = grad_real.float().pow(2).flatten(1).sum(1).mean()
    return penalty


def envelope_loss(y_real: torch.Tensor, y_fake: torch.Tensor,
                  window_sizes: List[int] = [64, 256, 1024]) -> torch.Tensor:
    """
    Multi-scale envelope loss — matches volume peaks and troughs at multiple
    temporal resolutions.

    Args:
        y_real: (B, 1, T) real waveform
        y_fake: (B, 1, T) generated waveform
        window_sizes: list of MaxPool kernel sizes for each scale
    Returns:
        Scalar loss (sum over scales)
    """
    loss = torch.tensor(0.0, device=y_real.device)
    for ws in window_sizes:
        stride = ws // 2
        pool = nn.MaxPool1d(kernel_size=ws, stride=stride)
        loss_pos = F.l1_loss(pool(y_real), pool(y_fake))
        loss_neg = F.l1_loss(pool(-y_real), pool(-y_fake))
        loss = loss + loss_pos + loss_neg
    return loss


# ---------------------------------------------------------------------------
# Multi-Resolution STFT Loss
# ---------------------------------------------------------------------------

@torch.amp.autocast("cuda", enabled=False)
def _stft_loss_one_scale(y: torch.Tensor, y_hat: torch.Tensor,
                         n_fft: int, hop_length: int,
                         win_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Spectral convergence + log-magnitude L1 at one STFT resolution.

    Forced to FP32 — torch.stft and log operations need full precision.
    """
    window = torch.hann_window(win_length, device=y.device)
    pad = (n_fft - hop_length) // 2

    y_pad = F.pad(y, (pad, pad), mode="reflect")
    y_hat_pad = F.pad(y_hat, (pad, pad), mode="reflect")

    spec_real = torch.stft(
        y_pad, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=False, return_complex=True,
    ).abs()
    spec_fake = torch.stft(
        y_hat_pad, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=False, return_complex=True,
    ).abs()

    # Spectral convergence: Frobenius-norm ratio
    sc = torch.norm(spec_real - spec_fake, p="fro") / (torch.norm(spec_real, p="fro") + _EPS)
    # Log-magnitude L1
    log_mag = F.l1_loss(
        torch.log(spec_real + _EPS),
        torch.log(spec_fake + _EPS),
    )
    return sc, log_mag


@torch.amp.autocast("cuda", enabled=False)
def multi_resolution_stft_loss(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    stft_configs: Optional[List[Tuple[int, int, int]]] = None,
) -> torch.Tensor:
    """
    Multi-resolution STFT loss (spectral convergence + log-magnitude L1).

    Args:
        y:     (B, 1, T) or (B, T) real waveform
        y_hat: (B, 1, T) or (B, T) generated waveform
        stft_configs: list of (n_fft, hop_length, win_length) tuples.
                      Default: 3 resolutions covering different time-frequency tradeoffs.
    Returns:
        Scalar loss (averaged over resolutions)
    """
    if stft_configs is None:
        stft_configs = [
            (2048, 512, 2048),
            (1024, 256, 1024),
            (512, 128, 512),
        ]

    if y.dim() == 3:
        y = y.squeeze(1)
    if y_hat.dim() == 3:
        y_hat = y_hat.squeeze(1)
    y = y.float()
    y_hat = y_hat.float()

    total_sc = torch.tensor(0.0, device=y.device)
    total_mag = torch.tensor(0.0, device=y.device)
    for n_fft, hop, win in stft_configs:
        sc, mag = _stft_loss_one_scale(y, y_hat, n_fft, hop, win)
        total_sc = total_sc + sc
        total_mag = total_mag + mag

    n = len(stft_configs)
    return (total_sc + total_mag) / n


# ---------------------------------------------------------------------------
# Phase Continuity Loss (APNet-style)
# ---------------------------------------------------------------------------

@torch.amp.autocast("cuda", enabled=False)
def phase_continuity_loss(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 480,
    win_length: int = 2048,
) -> torch.Tensor:
    """
    Phase continuity loss: penalize instantaneous-frequency and group-delay
    differences between real and generated waveforms.

    The iSTFT head predicts magnitude + phase explicitly, but unconstrained
    phase can have discontinuities across time and frequency that produce
    audible clicks/buzzing. This loss regularizes phase smoothness.

    Instantaneous frequency (IF) = d(phase)/dt across time frames
    Group delay (GD)             = d(phase)/df across frequency bins

    We compare IF and GD of real vs generated via L1, working on the
    wrapped-difference representation to handle 2pi wrapping correctly.

    APNet (Ai et al., 2023): "Phase anti-wrapping losses for phase continuity."

    Args:
        y:     (B, 1, T) or (B, T) real waveform
        y_hat: (B, 1, T) or (B, T) generated waveform
        n_fft:      STFT n_fft
        hop_length: STFT hop length
        win_length: STFT window length
    Returns:
        Scalar loss (IF + GD, averaged)
    """
    if y.dim() == 3:
        y = y.squeeze(1)
    if y_hat.dim() == 3:
        y_hat = y_hat.squeeze(1)
    y = y.float()
    y_hat = y_hat.float()

    window = torch.hann_window(win_length, device=y.device)
    pad = (n_fft - hop_length) // 2

    def _get_phase(x):
        x_pad = F.pad(x, (pad, pad), mode="reflect")
        spec = torch.stft(
            x_pad, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=False, return_complex=True,
        )
        return spec.angle()  # (B, n_bins, T_frames)

    phase_real = _get_phase(y)
    phase_fake = _get_phase(y_hat)

    # Instantaneous frequency: phase difference along time axis
    # Use wrapped difference: angle(exp(j*(phi[t+1] - phi[t])))
    # This correctly handles 2pi wrapping
    def _wrapped_diff(phase, dim):
        d = torch.diff(phase, n=1, dim=dim)
        # Wrap to [-pi, pi]
        return torch.atan2(torch.sin(d), torch.cos(d))

    if_real = _wrapped_diff(phase_real, dim=-1)   # (B, n_bins, T-1)
    if_fake = _wrapped_diff(phase_fake, dim=-1)
    loss_if = F.l1_loss(if_fake, if_real)

    # Group delay: phase difference along frequency axis
    gd_real = _wrapped_diff(phase_real, dim=-2)   # (B, n_bins-1, T)
    gd_fake = _wrapped_diff(phase_fake, dim=-2)
    loss_gd = F.l1_loss(gd_fake, gd_real)

    return (loss_if + loss_gd) / 2.0


# ---------------------------------------------------------------------------
# Mel Spectral Convergence Loss (operates on mel spectrograms directly)
# ---------------------------------------------------------------------------

@torch.amp.autocast("cuda", enabled=False)
def mel_spectral_convergence_loss(
    mel_real: torch.Tensor,
    mel_hat: torch.Tensor,
    x_mask: torch.Tensor,
) -> torch.Tensor:
    """Spectral convergence loss on mel spectrograms.

    Frobenius-norm ratio: ||mel_real - mel_hat||_F / ||mel_real||_F

    Complements L1: while L1 penalises absolute error uniformly,
    spectral convergence penalises relative error — regions with high
    energy (formants, harmonics) must be reproduced proportionally.
    Very cheap (~5 ops, no STFT), works directly on mel predictions.

    Args:
        mel_real: (B, n_mels, T) ground truth mel
        mel_hat:  (B, n_mels, T) predicted mel
        x_mask:   (B, 1, T) length mask
    Returns:
        Scalar loss
    """
    mel_real = mel_real.float()
    mel_hat = mel_hat.float()
    diff = (mel_real - mel_hat) * x_mask
    return torch.norm(diff, p="fro") / (torch.norm(mel_real * x_mask, p="fro") + _EPS)


# ---------------------------------------------------------------------------
# Gradient Balancer (EnCodec-style)
# ---------------------------------------------------------------------------

class GradientBalancer:
    """
    Automatic gradient balancing across multiple loss terms.

    Instead of manually tuning loss coefficients (c_mel, c_fm, etc.),
    the gradient balancer measures the gradient norm each loss produces
    on a reference parameter, tracks an EMA of these norms, and rescales
    gradients so each loss contributes according to its desired relative
    weight — regardless of magnitude differences.

    From Defossez et al., "High Fidelity Neural Audio Compression" (EnCodec).

    Usage:
        balancer = GradientBalancer(
            weights={"gen": 1.0, "fm": 1.0, "mel": 2.0, "env": 0.5, "mrstft": 1.0},
            ema_decay=0.999,
        )
        # In training loop:
        losses = {"gen": loss_gen, "fm": loss_fm, "mel": loss_mel, ...}
        balanced_loss = balancer.backward(losses, ref_param, scaler=scaler)
        # Then scaler.step(optimizer) as usual

    The returned scalar is for logging only — gradients are already
    accumulated on the parameters via backward().
    """

    def __init__(
        self,
        weights: Dict[str, float],
        ema_decay: float = 0.999,
    ):
        self.weights = weights
        self.ema_decay = ema_decay

        # EMA of per-loss gradient norms
        self._ema_norms: Dict[str, float] = {}
        self._initialized = False

    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        ref_param: torch.Tensor,
        scaler=None,
    ) -> torch.Tensor:
        """
        Compute balanced gradients and accumulate them.

        Args:
            losses:    dict of {name: scalar_loss} for each loss term
            ref_param: a reference parameter to measure gradient norms on
                       (any generator parameter works — we just need a
                        consistent measurement point)
            scaler:    GradScaler instance (for AMP). If provided, losses
                       are scaled before backward.

        Returns:
            total_loss: unbalanced weighted sum (for logging only)
        """
        # Step 1: Compute per-loss gradient norms on ref_param
        grad_norms = {}
        for name, loss in losses.items():
            if loss.requires_grad:
                # Compute gradient of this loss w.r.t. ref_param
                scaled = scaler.scale(loss) if scaler is not None else loss
                grad = torch.autograd.grad(
                    scaled, ref_param, retain_graph=True, allow_unused=True,
                )[0]
                if grad is not None:
                    # Unscale if using scaler
                    if scaler is not None:
                        grad = grad / scaler.get_scale()
                    grad_norms[name] = grad.detach().norm().item()
                else:
                    grad_norms[name] = 0.0
            else:
                grad_norms[name] = 0.0

        # Step 2: Update EMA of gradient norms
        if not self._initialized:
            for name in losses:
                self._ema_norms[name] = grad_norms.get(name, 1.0)
            self._initialized = True
        else:
            for name in losses:
                old = self._ema_norms.get(name, grad_norms.get(name, 1.0))
                new = grad_norms.get(name, 0.0)
                self._ema_norms[name] = self.ema_decay * old + (1 - self.ema_decay) * new

        # Step 3: Compute balanced coefficients
        # Target: each loss contributes gradient proportional to its weight
        # Scale factor for loss_i = weight_i / ema_norm_i (normalized)
        total_weight = sum(self.weights.get(n, 1.0) for n in losses)
        coeffs = {}
        for name in losses:
            w = self.weights.get(name, 1.0) / total_weight
            ema = max(self._ema_norms.get(name, 1.0), _EPS)
            coeffs[name] = w / ema

        # Normalize so the average coefficient is 1.0 (prevents gradient
        # scale from drifting, which would interact badly with GradScaler)
        avg_coeff = sum(coeffs.values()) / max(len(coeffs), 1)
        if avg_coeff > _EPS:
            for name in coeffs:
                coeffs[name] /= avg_coeff

        # Step 4: Compute balanced total loss
        balanced_loss = sum(
            coeffs[name] * losses[name]
            for name in losses
            if losses[name].requires_grad
        )

        # For logging: unbalanced weighted sum
        log_loss = sum(
            self.weights.get(name, 1.0) * losses[name].detach()
            for name in losses
        )

        # Backward the balanced loss
        if scaler is not None:
            scaler.scale(balanced_loss).backward()
        else:
            balanced_loss.backward()

        return log_loss
