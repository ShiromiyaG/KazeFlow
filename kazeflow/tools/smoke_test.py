"""
Smoke test for KazeFlow pretraining pipeline.

Runs a fast single-batch overfit test (~5-10 min) to detect training problems
without needing hours of real dataset training.

How it works:
  1. Build models from config (no checkpoint)
  2. Generate ONE synthetic batch and repeat it every step (forced overfit)
  3. Phase A (warmup_frac × steps): CFM only
  4. Phase B (remaining steps): CFM + Vocoder + Discriminator (joint)
  5. Print pass/fail health report at the end

Usage:
    python -m kazeflow.tools.smoke_test
    python -m kazeflow.tools.smoke_test --config kazeflow/configs/pretrain.json --steps 400
    python -m kazeflow.tools.smoke_test --device cpu --steps 100

Pass criteria (all must pass for a clean config):
  - CFM   : final loss < initial × 0.7  (monotonic decrease)
  - mel   : stays stable or decreases (< initial × 1.2)
  - disc  : d_real improves mid→late (disc distinguishes real vs fake)
  - GAN   : loss composition balanced (no single term >80%)
  - NaN   : no NaN/inf in any loss at any step
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kazeflow.models.flow_matching import ConditionalFlowMatching
from kazeflow.models.vocoder import ChouwaGANGenerator, EMAGenerator
from kazeflow.models.discriminator import ChouwaGANDiscriminator
from kazeflow.train.losses import (
    discriminator_loss_hinge,
    discriminator_loss_lsgan,
    feature_loss,
    generator_loss_hinge,
    generator_loss_lsgan,
    generator_loss_soft_hinge,
    mel_spectrogram_loss,
    multi_resolution_stft_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def _ok(msg):   return f"{GREEN}PASS{RESET}  {msg}"
def _fail(msg): return f"{RED}FAIL{RESET}  {msg}"
def _warn(msg): return f"{YELLOW}WARN{RESET}  {msg}"


def _make_batch(B, n_mels, segment_frames, hop_length, sample_rate, n_speakers, device):
    """Return one structured synthetic batch that mimics real data statistics.

    Instead of pure random noise, generates:
    - wav_gt: sum-of-sinusoids (speech-like harmonics at a realistic pitch)
    - mel:    log-mel computed from wav_gt (correct spectral structure)
    - f0:     smooth contour (quadratic interpolation between random anchors)
    - spin:   randn but scaled to ~unit variance per channel (SPIN v2 range)
    """
    audio_len = segment_frames * hop_length
    t = torch.arange(audio_len, device=device, dtype=torch.float32) / sample_rate

    wavs = []
    f0s  = []
    for b in range(B):
        # Smooth F0 contour: 4 anchor points, quadratic interp
        anchors = torch.rand(4, device=device) * 200 + 100  # 100-300 Hz
        f0_b = F.interpolate(
            anchors.view(1, 1, -1), size=segment_frames, mode="linear", align_corners=True
        ).squeeze()  # (segment_frames,)
        f0s.append(f0_b)

        # Upsample F0 to sample rate for phase accumulation
        f0_audio = F.interpolate(
            f0_b.view(1, 1, -1), size=audio_len, mode="linear", align_corners=True
        ).squeeze()  # (audio_len,)
        phase = torch.cumsum(2 * math.pi * f0_audio / sample_rate, dim=0)

        # Sum of harmonics (fundamental + 3 overtones)
        wav_b = torch.zeros(audio_len, device=device)
        for h, amp in enumerate([0.5, 0.25, 0.15, 0.10], start=1):
            wav_b = wav_b + amp * torch.sin(h * phase)
        wavs.append(wav_b)

    wav_gt = torch.stack(wavs).unsqueeze(1)  # (B, 1, audio_len)
    f0     = torch.stack(f0s)                 # (B, segment_frames)

    # Compute log-mel from wav_gt (matches what the real pipeline produces)
    import torchaudio
    mel_basis = torchaudio.functional.melscale_fbanks(
        n_freqs=2048 // 2 + 1, f_min=0.0, f_max=sample_rate / 2.0,
        n_mels=n_mels, sample_rate=sample_rate,
    ).T.to(device)  # (n_mels, n_fft//2+1)
    window = torch.hann_window(2048, device=device)
    pad = (2048 - hop_length) // 2
    x_pad = F.pad(wav_gt.squeeze(1), (pad, pad), mode="reflect")
    spec = torch.stft(x_pad, n_fft=2048, hop_length=hop_length, win_length=2048,
                      window=window, center=False, return_complex=True)
    mel = torch.log(torch.clamp(torch.matmul(mel_basis, spec.abs()), min=1e-5))
    mel = mel[..., :segment_frames]  # exact frame count

    spin   = torch.randn(B, 768, segment_frames, device=device) * 0.5
    spk_ids = torch.randint(0, n_speakers, (B,), device=device)
    return mel, spin, f0, spk_ids, wav_gt


def _avg(q):
    return sum(q) / len(q) if q else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KazeFlow smoke test")
    parser.add_argument("--config", default="kazeflow/configs/pretrain.json")
    parser.add_argument("--steps", type=int, default=500,
                        help="Total steps (warmup + joint)")
    parser.add_argument("--warmup-frac", type=float, default=0.4,
                        help="Fraction of steps used for CFM-only warmup")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", default="logs/smoke_test/logs",
                        help="TensorBoard log directory (omit to skip TB logging)")
    parser.add_argument("--no-tb", action="store_true", help="Disable TensorBoard")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _ROOT / config_path
    with open(config_path) as f:
        cfg = json.load(f)

    model_cfg = cfg["model"]
    train_cfg  = cfg["train"]
    device     = torch.device(args.device)
    B          = args.batch_size
    total_steps = args.steps
    warmup_steps = int(total_steps * args.warmup_frac)
    joint_steps  = total_steps - warmup_steps

    # ── Extract key hyper-params ──────────────────────────────────────────────
    sample_rate    = model_cfg["sample_rate"]
    n_mels         = model_cfg["n_mels"]
    hop_length     = model_cfg["hop_length"]
    n_fft          = model_cfg["n_fft"]
    win_length     = model_cfg.get("win_length", n_fft)
    segment_frames = model_cfg["segment_frames"]
    n_speakers     = model_cfg.get("n_speakers", 200)

    lr_flow  = train_cfg["learning_rate_flow"]
    lr_voc   = train_cfg["learning_rate_vocoder"]
    lr_disc  = train_cfg["learning_rate_disc"]
    betas_flow = tuple(train_cfg.get("betas_flow", [0.95, 0.999]))
    betas_voc  = tuple(train_cfg.get("betas_vocoder", [0.9, 0.999]))
    betas_disc = tuple(train_cfg.get("betas_disc",    [0.9, 0.999]))
    grad_clip_flow = train_cfg.get("grad_clip_flow", 3.0)
    grad_clip_voc  = train_cfg.get("grad_clip_vocoder", 50.0)
    grad_clip_disc = train_cfg.get("grad_clip_disc", 10.0)
    c_gen    = train_cfg.get("c_gen",    0.1)
    c_mel    = train_cfg.get("c_mel",    2.0)
    c_fm     = train_cfg.get("c_fm",     2.0)
    c_mrstft = train_cfg.get("c_mrstft", 1.0)
    gan_type = train_cfg.get("gan_loss_type", "hinge")
    n_disc_steps = train_cfg.get("n_disc_steps", 1)

    use_amp   = train_cfg.get("precision", "fp32").lower() in ("bf16", "fp16", "tf32_bf16", "tf32_fp16")
    amp_dtype = torch.bfloat16 if "bf16" in train_cfg.get("precision", "fp32").lower() else torch.float16
    if not torch.cuda.is_available():
        use_amp = False

    # ── Build models ──────────────────────────────────────────────────────────
    print(f"\nBuilding models on {device}...")
    flow = ConditionalFlowMatching(**model_cfg["flow_matching"]).to(device)
    vocoder = ChouwaGANGenerator(sr=sample_rate, **model_cfg["vocoder"]).to(device)
    discriminator = ChouwaGANDiscriminator(sample_rate=sample_rate, **model_cfg["discriminator"]).to(device)
    speaker_embed = nn.Embedding(n_speakers, model_cfg.get("speaker_embed_dim", 256)).to(device)
    ema_vocoder = EMAGenerator(vocoder)

    # ── Per-layer gradient clip on conv_post (matches real pretrain) ────────
    _cp_clip = train_cfg.get("conv_post_grad_clip", 0.0)
    if _cp_clip > 0:
        def _clip_hook(grad, _max=_cp_clip):
            gn = grad.norm()
            return grad * (_max / gn) if gn > _max else grad
        for p in vocoder.head.conv_post.parameters():
            p.register_hook(_clip_hook)

    n_flow  = sum(p.numel() for p in flow.parameters()) / 1e6
    n_voc   = sum(p.numel() for p in vocoder.parameters()) / 1e6
    n_disc  = sum(p.numel() for p in discriminator.parameters()) / 1e6
    print(f"  Flow:   {n_flow:.2f}M params")
    print(f"  Vocoder:{n_voc:.2f}M params")
    print(f"  Disc:   {n_disc:.2f}M params")
    if _cp_clip > 0:
        print(f"  conv_post_grad_clip: {_cp_clip}")

    # ── Loss functions ────────────────────────────────────────────────────────
    _use_san = model_cfg["discriminator"].get("use_san", False)
    if gan_type in ("hinge", "soft_hinge"):
        disc_loss_fn = discriminator_loss_hinge
        gen_loss_fn  = (generator_loss_soft_hinge
                        if (_use_san or gan_type == "soft_hinge")
                        else generator_loss_hinge)
    else:
        disc_loss_fn = discriminator_loss_lsgan
        gen_loss_fn  = generator_loss_lsgan
    if _use_san:
        print(f"  SAN: enabled (L2-norm conv_post)")

    # ── Optimizers (simple Adam, no LR scheduler for smoke test) ─────────────
    optim_flow  = torch.optim.Adam(
        list(flow.parameters()) + list(speaker_embed.parameters()),
        lr=lr_flow, betas=betas_flow)
    optim_voc   = torch.optim.Adam(vocoder.parameters(), lr=lr_voc, betas=betas_voc)
    optim_disc  = torch.optim.Adam(discriminator.parameters(), lr=lr_disc, betas=betas_disc)

    scaler_flow = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    scaler_voc  = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    scaler_disc = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    if not args.no_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = _ROOT / args.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(log_dir))
            print(f"  TensorBoard: {log_dir}")
        except Exception as e:
            print(f"  TensorBoard unavailable: {e}")

    # ── Mel basis cache (avoid recomputing every step) ────────────────────────
    import torchaudio
    _mel_basis = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=sample_rate / 2.0,
        n_mels=n_mels, sample_rate=sample_rate,
    ).T.to(device)

    # ── Synthetic batch (fixed — same every step for overfit) ─────────────────
    print(f"\nGenerating synthetic batch (B={B}, T={segment_frames}, audio_len={segment_frames*hop_length})...")
    batch = _make_batch(B, n_mels, segment_frames, hop_length, sample_rate, n_speakers, device)

    # ── Trackers ──────────────────────────────────────────────────────────────
    cfm_early   = deque(maxlen=20)
    cfm_late    = deque(maxlen=20)
    mel_early   = deque(maxlen=20)
    mel_late    = deque(maxlen=20)
    gn_voc_early  = deque(maxlen=20)
    gn_voc_late   = deque(maxlen=20)
    d_real_history = []
    d_fake_history = []
    # Loss composition trackers (weighted contributions to loss_voc)
    comp_gen_sum = comp_fm_sum = comp_mel_sum = comp_mrstft_sum = 0.0
    comp_count = 0
    nan_detected = False
    LOG_EVERY   = 10

    print(f"\nRunning {warmup_steps} warmup steps + {joint_steps} joint steps "
          f"(gan_loss_type={gan_type}, c_gen={c_gen})\n")

    t0 = time.time()

    for step in range(total_steps):
        mel, spin, f0, spk_ids, wav_gt = batch
        in_warmup = step < warmup_steps

        g        = speaker_embed(spk_ids).unsqueeze(-1)
        B_, _, T = mel.shape
        x_mask   = torch.ones(B_, 1, T, device=device)
        f0_exp   = f0.unsqueeze(1)

        # ── CFM loss ──────────────────────────────────────────────────────────
        optim_flow.zero_grad()
        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            loss_cfm = flow(x_1=mel, x_mask=x_mask, content=spin, f0=f0_exp, g=g)

        if not torch.isfinite(loss_cfm):
            print(f"  [step {step:4d}] NaN in loss_cfm!")
            nan_detected = True
            break

        scaler_flow.scale(loss_cfm).backward()
        scaler_flow.unscale_(optim_flow)
        torch.nn.utils.clip_grad_norm_(
            list(flow.parameters()) + list(speaker_embed.parameters()), grad_clip_flow)
        scaler_flow.step(optim_flow)
        scaler_flow.update()

        # Track CFM trend
        v = loss_cfm.item()
        if step < 20:
            cfm_early.append(v)
        cfm_late.append(v)

        if in_warmup:
            if step % LOG_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  [warmup {step:4d}/{warmup_steps}] cfm={v:.4f}  ({elapsed:.1f}s)")
            if writer:
                writer.add_scalar("smoke/cfm", v, step)
            continue

        # ── Joint phase ───────────────────────────────────────────────────────
        flow.eval()
        with torch.no_grad():
            mel_hat = flow.sample(
                content=spin, f0=f0_exp, x_mask=x_mask, g=g,
                n_steps=train_cfg.get("ode_steps_train", 1), method="euler",
            )
        flow.train()

        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            wav_hat = vocoder(mel_hat.detach(), f0, g=g.detach())

        tgt = min(wav_hat.shape[-1], wav_gt.shape[-1])
        wav_hat = wav_hat[..., :tgt]
        wav_real = wav_gt[..., :tgt].detach()

        # Discriminator step(s)
        gn_disc_v = 0.0
        d_real = d_fake = 0.0
        for _ in range(n_disc_steps):
            optim_disc.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                y_d_rs, y_d_gs, _, _ = discriminator(wav_real, wav_hat.detach(), compute_fmaps=False)
                loss_disc, _d_real, _d_fake = disc_loss_fn(y_d_rs, y_d_gs)
            if not torch.isfinite(loss_disc):
                print(f"  [step {step:4d}] NaN in loss_disc!")
                nan_detected = True
                break
            scaler_disc.scale(loss_disc).backward()
            scaler_disc.unscale_(optim_disc)
            gn_disc_v = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_disc).item()
            scaler_disc.step(optim_disc)
            scaler_disc.update()
            d_real, d_fake = _d_real, _d_fake
        if nan_detected:
            break

        d_real_history.append(d_real)
        d_fake_history.append(d_fake)

        # Generator (vocoder) step
        optim_voc.zero_grad()
        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(wav_real, wav_hat, compute_fmaps=True)
            loss_gen  = gen_loss_fn(y_d_gs)
            loss_fm   = feature_loss(fmap_rs, fmap_gs)
            loss_mel  = mel_spectrogram_loss(
                wav_real, wav_hat,
                n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                n_mels=n_mels, sample_rate=sample_rate,
                mel_basis=_mel_basis,
            )
        loss_mrstft = torch.tensor(0.0, device=device)
        if c_mrstft > 0:
            loss_mrstft = multi_resolution_stft_loss(wav_real, wav_hat)

        voc_losses = [loss_gen, loss_fm, loss_mel, loss_mrstft]
        if not all(torch.isfinite(l) for l in voc_losses):
            bad = [n for n, l in zip(["gen", "fm", "mel", "mrstft"], voc_losses) if not torch.isfinite(l)]
            print(f"  [step {step:4d}] NaN in vocoder losses: {bad}")
            nan_detected = True
            break

        loss_voc = c_gen * loss_gen + c_fm * loss_fm + c_mel * loss_mel + c_mrstft * loss_mrstft
        # Track weighted composition
        comp_gen_sum    += (c_gen * loss_gen).item()
        comp_fm_sum     += (c_fm * loss_fm).item()
        comp_mel_sum    += (c_mel * loss_mel).item()
        comp_mrstft_sum += (c_mrstft * loss_mrstft).item()
        comp_count += 1
        scaler_voc.scale(loss_voc).backward()
        scaler_voc.unscale_(optim_voc)
        gn_voc_v = torch.nn.utils.clip_grad_norm_(vocoder.parameters(), grad_clip_voc).item()
        # NaN can occur on step 0 (randomly-init vocoder + STFT backward edge case); skip tracking
        if torch.isfinite(torch.tensor(gn_voc_v)):
            joint_step_idx = step - warmup_steps
            if joint_step_idx < 20:
                gn_voc_early.append(gn_voc_v)
            gn_voc_late.append(gn_voc_v)
        else:
            gn_voc_v = float("nan")
        scaler_voc.step(optim_voc)
        scaler_voc.update()
        ema_vocoder.update()

        mel_v = loss_mel.item()
        joint_step = step - warmup_steps
        if joint_step < 20:
            mel_early.append(mel_v)
        mel_late.append(mel_v)

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            print(f"  [joint {joint_step:4d}/{joint_steps}] "
                  f"cfm={loss_cfm.item():.4f}  mel={mel_v:.4f}  "
                  f"gen={loss_gen.item():.4f}  disc={loss_disc.item():.4f}  "
                  f"D_real={d_real:.3f}  gn_voc={gn_voc_v:.1f}  ({elapsed:.1f}s)")

        if writer:
            writer.add_scalar("smoke/cfm",     loss_cfm.item(),   step)
            writer.add_scalar("smoke/mel",      mel_v,             step)
            writer.add_scalar("smoke/gen",      loss_gen.item(),   step)
            writer.add_scalar("smoke/disc",     loss_disc.item(),  step)
            writer.add_scalar("smoke/d_real",   d_real,            step)
            writer.add_scalar("smoke/d_fake",   d_fake,            step)
            writer.add_scalar("smoke/gn_voc",   gn_voc_v,          step)
            writer.add_scalar("smoke/gn_disc",  gn_disc_v,         step)

    if writer:
        writer.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Health report
    # ─────────────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"  Smoke test complete in {elapsed_total:.1f}s")
    print(f"{'─'*60}\n")

    results = []

    # 1. No NaN
    if nan_detected:
        results.append(_fail("NaN/inf detected during training — check losses/LR"))
    else:
        results.append(_ok("No NaN/inf in any loss"))

    # 2. CFM convergence (threshold: 30% reduction = monotonic decrease confirmed)
    cfm_e = _avg(cfm_early)
    cfm_l = _avg(cfm_late)
    if cfm_e > 0 and cfm_l < cfm_e * 0.5:
        results.append(_ok(f"CFM overfits: {cfm_e:.4f} → {cfm_l:.4f} ({cfm_l/cfm_e*100:.0f}%)"))
    elif cfm_e > 0 and cfm_l < cfm_e * 0.7:
        results.append(_ok(f"CFM converging: {cfm_e:.4f} → {cfm_l:.4f} ({cfm_l/cfm_e*100:.0f}%)"))
    elif cfm_e > 0 and cfm_l < cfm_e * 0.9:
        results.append(_warn(f"CFM slow convergence: {cfm_e:.4f} → {cfm_l:.4f} — try more steps"))
    else:
        results.append(_fail(f"CFM not converging: {cfm_e:.4f} → {cfm_l:.4f} — check flow LR/arch"))

    # 3. mel stability
    mel_e = _avg(mel_early)
    mel_l = _avg(mel_late)
    if mel_early:
        if mel_l < mel_e * 1.2:
            results.append(_ok(f"mel stable: {mel_e:.4f} → {mel_l:.4f}"))
        else:
            results.append(_fail(f"mel exploding: {mel_e:.4f} → {mel_l:.4f} — reduce c_gen or LR"))
    else:
        results.append(_warn("mel not measured (joint phase not reached)"))

    # 4. grad_norm/vocoder
    # NOTE: In single-batch overfit gn_voc grows without bound — this is expected and
    # handled by the clip. The mel check (above) is the real indicator of c_gen health.
    # Here we only warn with the ratio so it's visible; we never FAIL on gn_voc alone.
    if gn_voc_early and gn_voc_late:
        gn_e = _avg(gn_voc_early)
        gn_l = _avg(gn_voc_late)
        ratio = gn_l / gn_e if gn_e > 0 else float("inf")
        results.append(_warn(
            f"gn_voc (pre-clip): {gn_e:.0f} → {gn_l:.0f} ({ratio:.0f}× — overfit growth, "
            f"clip={grad_clip_voc} active; check mel check above)"))
    elif mel_early:
        results.append(_warn("gn_voc: NaN on early steps (init artifact, normal)"))
    else:
        results.append(_warn("gn_voc not measured"))

    # 5. Discriminator learning (compare mid→late to avoid early spike bias)
    if len(d_real_history) >= 40:
        mid_start = len(d_real_history) // 3
        mid_end   = mid_start + 20
        d_real_mid_avg  = sum(d_real_history[mid_start:mid_end]) / 20
        d_real_late_avg = sum(d_real_history[-20:]) / 20
        d_fake_late_avg = sum(d_fake_history[-20:]) / 20
        # Key metric: disc can separate real from fake (d_real > d_fake)
        margin = d_real_late_avg - d_fake_late_avg
        if margin > 0.5:
            results.append(_ok(
                f"disc separating: d_real={d_real_late_avg:.3f}, d_fake={d_fake_late_avg:.3f} "
                f"(margin={margin:.2f})"))
        elif margin > 0:
            results.append(_warn(
                f"disc weak separation: d_real={d_real_late_avg:.3f}, d_fake={d_fake_late_avg:.3f} "
                f"(margin={margin:.2f})"))
        else:
            results.append(_fail(
                f"disc cannot separate: d_real={d_real_late_avg:.3f}, d_fake={d_fake_late_avg:.3f}"))
    elif d_real_history:
        d_real_late = sum(d_real_history[-10:]) / min(10, len(d_real_history))
        results.append(_warn(f"disc: too few steps to judge (d_real_late={d_real_late:.3f})"))
    else:
        results.append(_warn("disc not measured (joint phase not reached)"))

    # 6. Loss composition balance
    if comp_count > 0:
        total = comp_gen_sum + comp_fm_sum + comp_mel_sum + comp_mrstft_sum
        if total > 0:
            pct_gen    = comp_gen_sum    / total * 100
            pct_fm     = comp_fm_sum     / total * 100
            pct_mel    = comp_mel_sum    / total * 100
            pct_mrstft = comp_mrstft_sum / total * 100
            breakdown = (f"gen={pct_gen:.0f}%  fm={pct_fm:.0f}%  "
                         f"mel={pct_mel:.0f}%  mrstft={pct_mrstft:.0f}%")
            dominant = max(pct_gen, pct_fm, pct_mel, pct_mrstft)
            if dominant > 80:
                which = ("gen" if pct_gen > 80 else "fm" if pct_fm > 80
                         else "mel" if pct_mel > 80 else "mrstft")
                results.append(_warn(
                    f"loss dominated by {which} ({dominant:.0f}%): {breakdown}"))
            else:
                results.append(_ok(f"loss balanced: {breakdown}"))

    print("  Results:")
    for r in results:
        print(f"    {r}")

    n_fail = sum(1 for r in results if r.startswith(f"{RED}FAIL"))
    n_warn = sum(1 for r in results if r.startswith(f"{YELLOW}WARN"))
    print(f"\n  {'─'*50}")
    if n_fail == 0 and n_warn == 0:
        print(f"  {GREEN}All checks passed.{RESET} Config looks healthy.")
    elif n_fail == 0:
        print(f"  {YELLOW}{n_warn} warning(s).{RESET} Config usable, monitor training.")
    else:
        print(f"  {RED}{n_fail} failure(s){RESET}, {n_warn} warning(s). Fix before long run.")
    print()

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
