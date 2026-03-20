"""
KazeFlow Training Pipeline.

Two-phase training:
1. Flow Matching: Train velocity field estimator (CFM loss on mel-spectrograms)
2. Vocoder (ChouwaGAN): Train mel → waveform with GAN losses

Both phases can run jointly or sequentially.

Precision options:
- FP32 (default): Standard full precision
- FP16 + AMP: Mixed precision with automatic scaling (faster, less VRAM)
- TF32: TensorFloat-32 on Ampere+ GPUs (faster matmuls, no accuracy loss)

torch.compile: Optional JIT compilation for ~10-30% speedup on PyTorch 2.0+.
"""

import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from kazeflow.models.flow_matching import ConditionalFlowMatching
from kazeflow.models.vocoder import ChouwaGANGenerator, EMAGenerator
from kazeflow.models.discriminator import ChouwaGANDiscriminator
from kazeflow.train.dataset import create_dataloader
from kazeflow.train.losses import (
    LeCamEMA,
    discriminator_loss_lsgan,
    discriminator_loss_hinge,
    feature_loss,
    generator_loss_lsgan,
    generator_loss_hinge,
    generator_loss_soft_hinge,
    mel_spectrogram_loss,
    multi_resolution_stft_loss,
    r1_gradient_penalty,
)

logger = logging.getLogger("kazeflow.train")


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """Convert a spectrogram to a NumPy RGB image array for TensorBoard."""
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return data


def _check_tf32_support() -> bool:
    """Check if GPU supports TF32 (Ampere / sm_80+)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 8  # sm_80+ (Ampere, Ada Lovelace, Hopper)


def _check_fp16_support() -> bool:
    """Check if GPU supports FP16 efficiently."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 7  # sm_70+ (Volta and above have tensor cores)


class KazeFlowTrainer:
    """
    Handles the full KazeFlow training loop:
    - Flow matching (CFM) for content→mel generation
    - ChouwaGAN vocoder for mel→waveform
    - ChouwaGAN discriminator for adversarial training

    Precision modes:
    - "fp32": Standard (default)
    - "fp16": FP16 mixed precision with AMP + GradScaler
    - "tf32": TF32 matmuls on Ampere+ (falls back to fp32 if unsupported)
    """

    def __init__(self, config: dict, output_dir: str, device: str = "cuda"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        model_cfg = config["model"]
        train_cfg = config["train"]

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # ── Precision setup ──────────────────────────────────────────────
        precision = train_cfg.get("precision", "fp32").lower()
        self.use_amp = False
        self.amp_dtype = torch.float32

        if precision == "bf16":
            # BF16: same dynamic range as FP32, no GradScaler needed.
            # Preferred over FP16 for training stability (no overflow at 65504).
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                logger.info("BF16 mixed precision enabled (recommended)")
            else:
                logger.warning("BF16 requested but not supported. Falling back to FP32.")
        elif precision == "tf32_fp16":
            # Combined: TF32 matmuls + FP16 AMP for maximum throughput
            if _check_tf32_support():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if _check_fp16_support():
                self.use_amp = True
                self.amp_dtype = torch.float16
            logger.info("TF32+FP16 combined precision enabled")
        elif precision == "tf32_bf16":
            tf32_ok = False
            if _check_tf32_support():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                tf32_ok = True
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                if tf32_ok:
                    logger.info("TF32+BF16 combined precision enabled (recommended)")
                else:
                    logger.info("BF16 enabled (GPU does not support TF32)")
            else:
                self.use_amp = True
                self.amp_dtype = torch.float16
                if tf32_ok:
                    logger.warning("BF16 not supported; falling back to TF32+FP16.")
                else:
                    logger.warning("Neither BF16 nor TF32 supported; falling back to FP16.")
        elif precision == "tf32":
            if _check_tf32_support():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("TF32 enabled (Ampere+ GPU detected)")
            else:
                logger.warning("TF32 requested but GPU doesn't support it (need sm_80+). Using FP32.")
        elif precision == "fp16":
            if _check_fp16_support():
                self.use_amp = True
                self.amp_dtype = torch.float16
                logger.info("FP16 mixed precision (AMP) enabled")
            else:
                logger.warning("FP16 requested but GPU doesn't support it efficiently. Using FP32.")
        elif precision == "fp32":
            logger.info("Using FP32 precision")
        else:
            logger.warning(f"Unknown precision '{precision}', falling back to FP32")

        # GradScaler is only needed for FP16 (BF16 has full FP32 dynamic range)
        self._needs_scaler = (self.use_amp and self.amp_dtype == torch.float16)

        # ── Models ───────────────────────────────────────────────────────
        self.flow = ConditionalFlowMatching(
            **model_cfg["flow_matching"]
        ).to(self.device)

        self.vocoder = ChouwaGANGenerator(
            sr=model_cfg["sample_rate"],
            **model_cfg["vocoder"]
        ).to(self.device)

        self.discriminator = ChouwaGANDiscriminator(
            sample_rate=model_cfg["sample_rate"],
            **model_cfg["discriminator"]
        ).to(self.device)

        # Speaker embedding
        n_speakers = model_cfg.get("n_speakers", 1)
        spk_dim = model_cfg.get("speaker_embed_dim", 256)
        self.speaker_embed = nn.Embedding(n_speakers, spk_dim).to(self.device)

        # ── Per-layer gradient clip on conv_post ─────────────────────────
        # conv_post (the final 1-d conv) receives ~97% of adversarial gradient
        # norm because it's closest to the discriminator.  Without per-layer
        # clipping it dominates the global norm, causing heavy global clip that
        # starves deeper layers of gradient signal.
        _cp_clip = train_cfg.get("conv_post_grad_clip", 0.0)
        if _cp_clip > 0:
            def _clip_hook(grad, _max=_cp_clip):
                gn = grad.norm()
                return grad * (_max / gn) if gn > _max else grad
            for p in self.vocoder.head.conv_post.parameters():
                p.register_hook(_clip_hook)
            logger.info(f"conv_post per-layer gradient clip: {_cp_clip}")

        # ── torch.compile ────────────────────────────────────────────────
        use_compile = train_cfg.get("torch_compile", False)
        if use_compile:
            compile_mode = train_cfg.get("compile_mode", "default")
            compile_disc = train_cfg.get("compile_disc", False)
            try:
                self.flow.estimator = torch.compile(
                    self.flow.estimator, mode=compile_mode)
                self.vocoder.head = torch.compile(
                    self.vocoder.head, mode=compile_mode)
                if compile_disc:
                    self.discriminator = torch.compile(
                        self.discriminator, mode=compile_mode)
                    logger.info(f"torch.compile applied (flow + vocoder + discriminator, mode='{compile_mode}')")
                else:
                    logger.info(f"torch.compile applied (flow + vocoder, mode='{compile_mode}'; discriminator skipped)")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")

        # ── Optimizers ───────────────────────────────────────────────────
        lr_flow = train_cfg["learning_rate_flow"]
        lr_voc = train_cfg["learning_rate_vocoder"]
        lr_disc = train_cfg["learning_rate_disc"]
        betas = tuple(train_cfg["betas"])
        betas_flow = tuple(train_cfg.get("betas_flow", [0.95, 0.999]))
        wd = train_cfg.get("weight_decay", 0.01)
        fused = torch.cuda.is_available()

        # Split flow parameters: exclude biases, norms, and zero-init
        # out_proj from weight decay.
        flow_decay, flow_no_decay = [], []
        for name, p in list(self.flow.named_parameters()) + \
                        [(f"spk.{n}", p) for n, p in self.speaker_embed.named_parameters()]:
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "norm" in name or "out_proj" in name:
                flow_no_decay.append(p)
            else:
                flow_decay.append(p)
        self.optim_flow = torch.optim.AdamW([
            {"params": flow_decay, "weight_decay": wd},
            {"params": flow_no_decay, "weight_decay": 0.0},
        ], lr=lr_flow, betas=betas_flow, fused=fused)
        betas_vocoder = tuple(train_cfg.get("betas_vocoder", [0.9, 0.999]))
        betas_disc = tuple(train_cfg.get("betas_disc", [0.9, 0.999]))
        self.optim_vocoder = torch.optim.AdamW(
            self.vocoder.parameters(),
            lr=lr_voc, betas=betas_vocoder, weight_decay=wd, fused=fused,
        )
        # Selective weight decay for discriminator:
        # - Spectral norm: wd=0 everywhere (SN renormalizes, WD is wasted)
        # - SAN: wd>0 for body (weight_norm magnitude `g` drifts without it),
        #        wd=0 for conv_post (L2-normed by SAN, WD fights parametrization)
        # - Otherwise: wd=wd everywhere
        _disc_cfg = model_cfg.get("discriminator", {})
        if _disc_cfg.get("use_spectral_norm", False):
            self.optim_disc = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=lr_disc, betas=betas_disc, weight_decay=0.0, fused=fused,
            )
        else:
            disc_body, disc_conv_post = [], []
            for name, p in self.discriminator.named_parameters():
                if not p.requires_grad:
                    continue
                if "conv_post" in name:
                    disc_conv_post.append(p)
                else:
                    disc_body.append(p)
            self.optim_disc = torch.optim.AdamW([
                {"params": disc_body, "weight_decay": wd},
                {"params": disc_conv_post, "weight_decay": 0.0},
            ], lr=lr_disc, betas=betas_disc, fused=fused)

        # ── Schedulers ───────────────────────────────────────────────────
        lr_decay = train_cfg.get("lr_decay", 0.999)
        self.sched_flow = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_flow, gamma=lr_decay)
        self.sched_vocoder = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_vocoder, gamma=lr_decay)
        self.sched_disc = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_disc, gamma=lr_decay)

        # ── AMP GradScalers (only active with FP16) ────────────────────
        # CRITICAL: Separate scalers for generator path vs discriminator.
        # A single shared scaler causes cross-contamination: when the disc
        # hits inf gradients the scale reduction affects the generator too,
        # and vice versa. This interaction is the primary cause of NaN
        # after ~100k steps in adversarial training with FP16.
        self.scaler_flow = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)
        self.scaler_gen = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)
        self.scaler_disc = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)

        # ── Logging ──────────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        self.global_step = 0
        self.epoch = 0

        # ── Config shortcuts ─────────────────────────────────────────────
        self.sample_rate = model_cfg["sample_rate"]
        self.hop_length = model_cfg["hop_length"]
        self.n_mels = model_cfg["n_mels"]
        self.segment_frames = model_cfg["segment_frames"]
        self.c_mel = train_cfg["c_mel"]
        self.c_fm = train_cfg["c_fm"]
        self.c_gen = train_cfg.get("c_gen", 1.0)
        self.c_mrstft = train_cfg.get("c_mrstft", 0.0)
        self.c_r1 = train_cfg.get("c_r1", 0.0)
        self.r1_interval = train_cfg.get("r1_interval", 4)
        self.n_disc_steps = max(1, int(train_cfg.get("n_disc_steps", 1)))

        # GAN loss type: "hinge" | "soft_hinge" | "lsgan"
        #   hinge:      disc=hinge_margin, gen=hinge_linear  (BigVGAN style)
        #   soft_hinge: disc=hinge_margin, gen=softplus(-D)  (StyleGAN2-style gen smoothing)
        #   lsgan:      disc=lsgan,        gen=lsgan
        _gan_type = train_cfg.get("gan_loss_type", "hinge")

        if _gan_type in ("hinge", "soft_hinge"):
            self._disc_loss_fn = discriminator_loss_hinge
            self._gen_loss_fn = (
                generator_loss_soft_hinge if _gan_type == "soft_hinge"
                else generator_loss_hinge
            )
        else:
            self._disc_loss_fn = discriminator_loss_lsgan
            self._gen_loss_fn = generator_loss_lsgan

        logger.info(f"GAN loss: {_gan_type}")
        self.grad_clip_flow = train_cfg["grad_clip_flow"]
        self.grad_clip_voc = train_cfg["grad_clip_vocoder"]
        self.grad_clip_disc = train_cfg["grad_clip_disc"]
        self.c_lecam = train_cfg.get("c_lecam", 0.0)
        self.lecam = LeCamEMA(decay=train_cfg.get("lecam_ema_decay", 0.999)) if self.c_lecam > 0 else None
        self.cfm_grad_accum = max(1, int(train_cfg.get("cfm_grad_accum", 1)))

        # Mel basis cache
        self._mel_basis = None

        # Reference sample for eval inference (lazy-initialized)
        self._reference = None

        # ── EMA Generator ────────────────────────────────────────────────
        ema_decay = train_cfg.get("ema_decay", 0.999)
        self.ema_vocoder = EMAGenerator(self.vocoder, decay=ema_decay)
        self.ema_vocoder.to(self.device)
        logger.info(f"EMA vocoder initialized (decay={ema_decay})")



    def _eval_infer(self, mel_ref, spin_ref, f0_ref, spk_id_ref):
        """
        Run the full generation pipeline in eval mode on a reference sample.
        Uses EMA vocoder for higher-quality inference.

        Returns:
            mel_hat: (1, n_mels, T) generated mel-spectrogram
            wav_hat: (1, 1, T_audio) generated waveform
        """
        self.flow.eval()
        ema_voc = self.ema_vocoder.get_model()
        ema_voc.eval()
        with torch.no_grad():
            g = self.speaker_embed(spk_id_ref).unsqueeze(-1)
            B, _, T = mel_ref.shape
            x_mask = torch.ones(B, 1, T, device=self.device)
            f0_expanded = f0_ref.unsqueeze(1)  # (B, 1, T)

            mel_hat = self.flow.sample(
                content=spin_ref,
                f0=f0_expanded,
                x_mask=x_mask,
                g=g,
                n_steps=self.config["train"].get("ode_steps_infer", 16),
                method="euler",
            )

            wav_hat = ema_voc(mel_hat, f0_ref, g=g)
        self.flow.train()
        return mel_hat, wav_hat

    def _get_reference_sample(self, dataloader):
        """Get the first batch sample as reference for eval inference."""
        if self._reference is not None:
            return self._reference

        batch = next(iter(dataloader))
        mel, spin, f0, spk_ids, _wav_gt = [x.to(self.device) for x in batch]
        # Use only the first sample in the batch
        self._reference = (
            mel[:1],
            spin[:1],
            f0[:1],
            spk_ids[:1],
        )
        return self._reference

    def _get_mel_basis(self):
        if self._mel_basis is None:
            import torchaudio
            self._mel_basis = torchaudio.functional.melscale_fbanks(
                n_freqs=self.config["model"]["n_fft"] // 2 + 1,
                f_min=0.0, f_max=self.sample_rate / 2.0,
                n_mels=self.n_mels, sample_rate=self.sample_rate,
            ).T.to(self.device)
        return self._mel_basis

    def train(self, filelist_path: str, dataset_root: str,
              resume_path: str = None, stop_event=None):
        """Main training loop.

        Args:
            stop_event: Optional ``threading.Event``. When set, the loop exits
                cleanly after the current epoch completes and saves a checkpoint.
        """
        train_cfg = self.config["train"]

        if resume_path:
            self._load_checkpoint(resume_path)

        dataloader = create_dataloader(
            filelist_path=filelist_path,
            dataset_root=dataset_root,
            batch_size=train_cfg["batch_size"],
            segment_frames=self.segment_frames,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            num_workers=train_cfg["num_workers"],
            pin_memory=train_cfg.get("pin_memory", True),
        )

        epochs = train_cfg["epochs"]
        save_every = train_cfg["save_every"]
        log_every = train_cfg["log_every"]

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Dataset: {len(dataloader.dataset)} samples")
        logger.info(f"Batch size: {train_cfg['batch_size']}")


        for epoch in range(self.epoch, epochs):
            # Check stop signal before starting a new epoch
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop signal received — exiting training loop.")
                break

            self.epoch = epoch
            self.flow.train()
            self.vocoder.train()
            self.discriminator.train()

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )
            _accum_cfm = 0
            _gn_flow = torch.tensor(0.0)
            for batch_idx, batch in enumerate(pbar):
                mel, spin, f0, spk_ids, wav_gt = [x.to(self.device) for x in batch]

                # Speaker embedding: (B, gin_ch, 1)
                g = self.speaker_embed(spk_ids).unsqueeze(-1)

                # Mask: all ones for fixed-length segments
                B, _, T = mel.shape
                x_mask = torch.ones(B, 1, T, device=self.device)
                f0_expanded = f0.unsqueeze(1)  # (B, 1, T)

                # ── Step 1: Flow Matching ────────────────────────────
                if _accum_cfm == 0:
                    self.optim_flow.zero_grad()

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    loss_cfm = self.flow(
                        x_1=mel, x_mask=x_mask,
                        content=spin, f0=f0_expanded, g=g,
                    )

                # NaN guard: reset accumulation and skip
                if not torch.isfinite(loss_cfm):
                    logger.warning(
                        f"Step {self.global_step}: loss_cfm is "
                        f"{loss_cfm.item()}, skipping step"
                    )
                    self.optim_flow.zero_grad(set_to_none=True)
                    _accum_cfm = 0
                    self.global_step += 1
                    continue

                self.scaler_flow.scale(loss_cfm / self.cfm_grad_accum).backward()
                _accum_cfm += 1
                _do_flow_step = (_accum_cfm == self.cfm_grad_accum)

                if _do_flow_step:
                    self.scaler_flow.unscale_(self.optim_flow)
                    _gn_flow = torch.nn.utils.clip_grad_norm_(
                        self.flow.parameters(), self.grad_clip_flow)
                    self.scaler_flow.step(self.optim_flow)
                    _accum_cfm = 0

                # ── Step 2: Generate mel from flow (for vocoder) ─────
                self.flow.eval()
                with torch.no_grad():
                    mel_hat = self.flow.sample(
                        content=spin, f0=f0_expanded,
                        x_mask=x_mask, g=g,
                        n_steps=self.config["train"].get("ode_steps_train", 1),
                        method="euler",
                    )
                self.flow.train()

                # ── Step 3: Vocoder (mel → waveform) ─────────────────
                # Vocoder forward — only on generated mel (single forward)
                # wav_gt from dataset provides the real audio directly,
                # avoiding a second vocoder forward that would double VRAM.
                # g.detach(): speaker_embed graph was freed by CFM backward;
                # vocoder loss should not update speaker_embed (it's in optim_flow).
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    wav_hat = self.vocoder(mel_hat.detach(), f0, g=g.detach())

                    # Align lengths (vocoder iSTFT output may differ slightly)
                    target_len = min(wav_hat.shape[-1], wav_gt.shape[-1])
                    wav_hat = wav_hat[..., :target_len]
                    wav_real_detached = wav_gt[..., :target_len].detach()

                # ── Step 4: Discriminator — repeated n_disc_steps times per gen step
                gn_disc = None
                d_real_mean = d_fake_mean = 0.0
                for _disc_i in range(self.n_disc_steps):
                    self.optim_disc.zero_grad()
                    # R1 gradient penalty needs grad w.r.t. real audio
                    _apply_r1 = (self.c_r1 > 0
                                 and self.global_step % self.r1_interval == 0)
                    if _apply_r1:
                        wav_real_detached.requires_grad_(True)

                    with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                        y_d_rs, y_d_gs, _, _ = self.discriminator(
                            wav_real_detached, wav_hat.detach(),
                            compute_fmaps=False,
                        )
                        loss_disc, _d_real, _d_fake = self._disc_loss_fn(
                            y_d_rs, y_d_gs)

                    # LeCam regularization (prevents overconfident disc)
                    loss_lecam = torch.tensor(0.0, device=self.device)
                    if self.lecam is not None:
                        loss_lecam = self.lecam.penalty(y_d_rs, y_d_gs)
                        self.lecam.update(y_d_rs, y_d_gs)
                        loss_disc = loss_disc + self.c_lecam * loss_lecam

                    # R1 penalty (FP32, outside autocast)
                    if _apply_r1:
                        r1_pen = r1_gradient_penalty(y_d_rs, wav_real_detached)
                        loss_disc = loss_disc + (self.c_r1 / 2) * self.r1_interval * r1_pen
                        wav_real_detached.requires_grad_(False)

                    # NaN guard: skip disc step if loss is bad
                    if not torch.isfinite(loss_disc):
                        logger.warning(
                            f"Step {self.global_step}: loss_disc is "
                            f"{loss_disc.item()}, skipping disc step"
                        )
                        self.optim_disc.zero_grad(set_to_none=True)
                    else:
                        self.scaler_disc.scale(loss_disc).backward()
                        self.scaler_disc.unscale_(self.optim_disc)
                        gn_disc = torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), self.grad_clip_disc)
                        self.scaler_disc.step(self.optim_disc)
                        self.scaler_disc.update()
                    # Keep scores of last disc substep for logging
                    d_real_mean, d_fake_mean = _d_real, _d_fake

                # ── Step 5: Generator (vocoder) losses ───────────────
                self.optim_vocoder.zero_grad()

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(
                        wav_real_detached, wav_hat, compute_fmaps=True,
                    )
                    loss_gen = self._gen_loss_fn(y_d_gs)
                    loss_fm = feature_loss(fmap_rs, fmap_gs)
                    loss_mel = mel_spectrogram_loss(
                        wav_real_detached, wav_hat,
                        n_fft=self.config["model"]["n_fft"],
                        hop_length=self.hop_length,
                        win_length=self.config["model"]["win_length"],
                        n_mels=self.n_mels,
                        sample_rate=self.sample_rate,
                        mel_basis=self._get_mel_basis(),
                    )

                loss_mrstft = torch.tensor(0.0, device=self.device)
                if self.c_mrstft > 0:
                    loss_mrstft = multi_resolution_stft_loss(
                        wav_real_detached, wav_hat)

                # NaN guard for vocoder losses
                voc_losses = [loss_gen, loss_fm, loss_mel, loss_mrstft]
                if not all(torch.isfinite(l) for l in voc_losses):
                    bad = [n for n, l in zip(
                        ["gen", "fm", "mel", "mrstft"], voc_losses
                    ) if not torch.isfinite(l)]
                    logger.warning(
                        f"Step {self.global_step}: NaN/inf in vocoder losses "
                        f"{bad}, skipping vocoder step"
                    )
                    self.optim_vocoder.zero_grad(set_to_none=True)
                    gn_voc = None
                else:
                    loss_voc_total = (
                        self.c_gen * loss_gen
                        + self.c_fm * loss_fm
                        + self.c_mel * loss_mel
                        + self.c_mrstft * loss_mrstft
                    )
                    self.scaler_gen.scale(loss_voc_total).backward()
                    self.scaler_gen.unscale_(self.optim_vocoder)
                    gn_voc = torch.nn.utils.clip_grad_norm_(
                        self.vocoder.parameters(), self.grad_clip_voc)
                    self.scaler_gen.step(self.optim_vocoder)
                    self.ema_vocoder.update()

                if _do_flow_step:
                    self.scaler_flow.update()
                self.scaler_gen.update()
                self.global_step += 1

                # ── Logging ──────────────────────────────────────────
                pbar.set_postfix(
                    cfm=f"{loss_cfm.item():.4f}",
                    mel=f"{loss_mel.item():.4f}",
                    gen=f"{loss_gen.item():.4f}",
                    disc=f"{loss_disc.item():.4f}",
                )

                if self.global_step % log_every == 0:
                    step = self.global_step
                    # Losses
                    self.writer.add_scalar("loss/cfm", loss_cfm.item(), step)
                    self.writer.add_scalar("loss/disc", loss_disc.item(), step)
                    self.writer.add_scalar("loss/gen", loss_gen.item(), step)
                    self.writer.add_scalar("loss/fm", loss_fm.item(), step)
                    self.writer.add_scalar("loss/mel", loss_mel.item(), step)
                    if self.c_mrstft > 0:
                        self.writer.add_scalar("loss/mrstft", loss_mrstft.item(), step)
                    if gn_voc is not None:  # loss_voc_total only exists when step succeeded
                        self.writer.add_scalar("loss/voc_total", loss_voc_total.item(), step)
                    # Discriminator health
                    if gn_disc is not None:  # d_real/d_fake valid when disc step succeeded
                        self.writer.add_scalar("d_score/real", d_real_mean, step)
                        self.writer.add_scalar("d_score/fake", d_fake_mean, step)
                    # Gradient norms (detect explosions / vanishing)
                    self.writer.add_scalar("grad_norm/flow", _gn_flow.item(), step)
                    if gn_disc is not None:
                        self.writer.add_scalar("grad_norm/disc", gn_disc.item(), step)
                    if gn_voc is not None:
                        self.writer.add_scalar("grad_norm/vocoder", gn_voc.item(), step)
                    # Learning rates
                    self.writer.add_scalar("lr/flow", self.optim_flow.param_groups[0]["lr"], step)
                    self.writer.add_scalar("lr/vocoder", self.optim_vocoder.param_groups[0]["lr"], step)
                    self.writer.add_scalar("lr/disc", self.optim_disc.param_groups[0]["lr"], step)
                    # R1 gradient penalty
                    if _apply_r1:
                        self.writer.add_scalar("loss/r1", r1_pen.item(), step)
                    # LeCam regularization
                    if self.lecam is not None:
                        self.writer.add_scalar("loss/lecam", loss_lecam.item(), step)

                if stop_event is not None and stop_event.is_set():
                    break

            # ── End of epoch ─────────────────────────────────────────
            # Propagate inner stop-event break to the epoch loop
            if stop_event is not None and stop_event.is_set():
                break

            self.sched_flow.step()
            self.sched_vocoder.step()
            self.sched_disc.step()

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch + 1)

                # ── Eval inference → TensorBoard ─────────────────────
                try:
                    ref_mel, ref_spin, ref_f0, ref_spk = self._get_reference_sample(dataloader)
                    mel_hat_eval, wav_hat_eval = self._eval_infer(
                        ref_mel, ref_spin, ref_f0, ref_spk,
                    )

                    mel_orig_np = ref_mel[0].detach().cpu().float().numpy()
                    mel_gen_np = mel_hat_eval[0].detach().cpu().float().numpy()

                    # Log spectrogram images
                    self.writer.add_image(
                        "eval/mel_original",
                        plot_spectrogram_to_numpy(mel_orig_np),
                        self.global_step,
                        dataformats="HWC",
                    )
                    self.writer.add_image(
                        "eval/mel_generated",
                        plot_spectrogram_to_numpy(mel_gen_np),
                        self.global_step,
                        dataformats="HWC",
                    )
                    # Difference map (brighter = larger error)
                    self.writer.add_image(
                        "eval/mel_diff",
                        plot_spectrogram_to_numpy(
                            np.abs(mel_orig_np - mel_gen_np)
                        ),
                        self.global_step,
                        dataformats="HWC",
                    )

                    # Log generated audio
                    self.writer.add_audio(
                        "eval/audio",
                        wav_hat_eval[0, :, :].detach().cpu().float(),
                        self.global_step,
                        sample_rate=self.sample_rate,
                    )

                    self.writer.flush()
                    logger.info(
                        f"Eval inference logged at epoch {epoch+1}, "
                        f"step {self.global_step}"
                    )
                except Exception as e:
                    logger.warning(f"Eval inference failed: {e}")

        stopped_early = stop_event is not None and stop_event.is_set()
        self._save_checkpoint(self.epoch if stopped_early else epochs)
        if stopped_early:
            logger.info(
                "Training stopped by user at epoch %d. Checkpoint saved.",
                self.epoch,
            )
        else:
            logger.info("Training complete.")
        self._cleanup()

    def _cleanup(self):
        """Close TensorBoard writer and release GPU memory."""
        try:
            self.writer.close()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory released.")

    def __del__(self):
        try:
            self._cleanup()
        except Exception:
            pass

    @staticmethod
    def _unwrap_model(model):
        """Unwrap DDP and torch.compile wrappers to get the raw module."""
        if hasattr(model, "module"):        # DistributedDataParallel
            model = model.module
        if hasattr(model, "_orig_mod"):     # torch.compile (OptimizedModule)
            model = model._orig_mod
        return model

    @staticmethod
    def _normalize_checkpoint_sd(state_dict: dict) -> dict:
        """
        Return a CLEAN copy of a state dict — all ``_orig_mod.`` prefixes stripped,
        derived buffers dropped.  Always call this before saving or before
        _align_state_dict so every checkpoint is format-agnostic.
        """
        # Strip top-level _orig_mod. prefix (entire model compiled)
        if state_dict and all(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        # Strip _orig_mod. as a sub-module infix (individual sub-module compiled)
        state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        _DERIVED_SUFFIXES = ("_shared_window",)
        return {
            k: v for k, v in state_dict.items()
            if not any(k.endswith(s) for s in _DERIVED_SUFFIXES)
        }

    @staticmethod
    def _align_state_dict(clean_sd: dict, model) -> dict:
        """
        Remap CLEAN checkpoint keys so they exactly match the live model's
        state_dict layout.

        When sub-modules are torch.compiled the model's own state_dict has
        ``._orig_mod.`` infixes (e.g. ``estimator._orig_mod.weight``).  A
        checkpoint normalised by _normalize_checkpoint_sd has clean keys
        (``estimator.weight``).  This function re-introduces the infixes so
        load_state_dict gets an exact match regardless of whether compile was
        active when the checkpoint was saved.
        """
        def _clean(k): return k.replace("._orig_mod.", ".")
        clean_to_model = {_clean(k): k for k in model.state_dict()}
        return {clean_to_model.get(_clean(k), k): v for k, v in clean_sd.items()}

    def _save_checkpoint(self, epoch: int):
        path = self.output_dir / f"checkpoint_{epoch}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "flow": self._normalize_checkpoint_sd(self._unwrap_model(self.flow).state_dict()),
            "vocoder": self._normalize_checkpoint_sd(self._unwrap_model(self.vocoder).state_dict()),
            "vocoder_ema": self._normalize_checkpoint_sd(self.ema_vocoder.state_dict()),
            "discriminator": self._normalize_checkpoint_sd(self._unwrap_model(self.discriminator).state_dict()),
            "speaker_embed": self._unwrap_model(self.speaker_embed).state_dict(),
            "scaler_flow": self.scaler_flow.state_dict(),
            "scaler_gen": self.scaler_gen.state_dict(),
            "scaler_disc": self.scaler_disc.state_dict(),
            "optim_flow": self.optim_flow.state_dict(),
            "optim_vocoder": self.optim_vocoder.state_dict(),
            "optim_disc": self.optim_disc.state_dict(),
            "sched_flow": self.sched_flow.state_dict(),
            "sched_vocoder": self.sched_vocoder.state_dict(),
            "sched_disc": self.sched_disc.state_dict(),
        }, tmp_path)
        tmp_path.replace(path)
        logger.info(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        flow = self._unwrap_model(self.flow)
        flow.load_state_dict(
            self._align_state_dict(self._normalize_checkpoint_sd(ckpt["flow"]), flow))
        vocoder = self._unwrap_model(self.vocoder)
        vocoder.load_state_dict(
            self._align_state_dict(self._normalize_checkpoint_sd(ckpt["vocoder"]), vocoder))
        if "vocoder_ema" in ckpt:
            self.ema_vocoder.load_state_dict(
                self._align_state_dict(self._normalize_checkpoint_sd(ckpt["vocoder_ema"]), self.ema_vocoder))
        else:
            logger.warning("No EMA state in checkpoint — EMA initialized from vocoder weights")
        disc = self._unwrap_model(self.discriminator)
        disc.load_state_dict(
            self._align_state_dict(self._normalize_checkpoint_sd(ckpt["discriminator"]), disc),
            strict=False)
        spk = self._unwrap_model(self.speaker_embed)
        spk.load_state_dict(
            self._normalize_checkpoint_sd(ckpt["speaker_embed"]))
        self.optim_flow.load_state_dict(ckpt["optim_flow"])
        self.optim_vocoder.load_state_dict(ckpt["optim_vocoder"])
        try:
            self.optim_disc.load_state_dict(ckpt["optim_disc"])
        except ValueError as e:
            # Param group count changed (e.g. 1→2 after selective weight_decay fix).
            # Restore per-tensor moments only; param_groups are already correct.
            logger.warning(
                f"optim_disc param group mismatch ({e}); "
                "restoring per-tensor moments only (param_groups kept as-is)."
            )
            saved_state = ckpt["optim_disc"].get("state", {})
            for param_id, s in saved_state.items():
                if param_id in self.optim_disc.state:
                    self.optim_disc.state[param_id].update(s)
                else:
                    self.optim_disc.state[param_id] = s
        self.sched_flow.load_state_dict(ckpt["sched_flow"])
        self.sched_vocoder.load_state_dict(ckpt["sched_vocoder"])
        self.sched_disc.load_state_dict(ckpt["sched_disc"])
        # Override LRs from config — load_state_dict restores the checkpoint's
        # param_groups (including the old LR), so any config changes only take
        # effect if we explicitly re-apply them here. We rescale the scheduler's
        # base_lrs and _last_lr by the same ratio so the decay curve stays consistent.
        train_cfg = self.config["train"]
        for optim, sched, new_base in [
            (self.optim_flow,    self.sched_flow,    train_cfg["learning_rate_flow"]),
            (self.optim_vocoder, self.sched_vocoder, train_cfg["learning_rate_vocoder"]),
            (self.optim_disc,    self.sched_disc,    train_cfg["learning_rate_disc"]),
        ]:
            ckpt_base = sched.base_lrs[0]
            if abs(ckpt_base - new_base) > 1e-12:
                ratio = new_base / ckpt_base
                sched.base_lrs = [lr * ratio for lr in sched.base_lrs]
                sched._last_lr  = [lr * ratio for lr in sched._last_lr]
                for pg in optim.param_groups:
                    pg["lr"] *= ratio
                logger.info(
                    f"LR override: {ckpt_base:.2e} → {new_base:.2e} (ratio {ratio:.4f})"
                )
        # Re-apply selective weight_decay for disc after checkpoint load.
        # With SAN: body gets wd>0 (group 0), conv_post gets wd=0 (group 1).
        # With SN: all groups get wd=0.
        _use_sn = self.config["model"].get("discriminator", {}).get("use_spectral_norm", False)
        _wd = train_cfg.get("weight_decay", 0.01)
        for i, pg in enumerate(self.optim_disc.param_groups):
            # group 0 = body (wd>0 unless SN), group 1 = conv_post (always wd=0)
            target_wd = 0.0 if (_use_sn or i == 1) else _wd
            if abs(pg.get("weight_decay", 0.0) - target_wd) > 1e-8:
                logger.info(
                    f"disc param_group[{i}] weight_decay override: "
                    f"{pg['weight_decay']} → {target_wd}"
                )
            pg["weight_decay"] = target_wd
        # Optionally reset Adam moments for disc + vocoder (without touching model
        # weights or LR).  Useful when resuming with significantly changed
        # adversarial hyperparameters (e.g. c_gen, lr_vocoder) so that stale
        # momentum built up under the old regime doesn't shock the new one.
        # Set "reset_gan_optimizers": true in config for one resume, then flip
        # back to false to avoid re-resetting on every subsequent resume.
        if train_cfg.get("reset_gan_optimizers", False):
            for optim, name in [(self.optim_disc, "disc"), (self.optim_vocoder, "vocoder")]:
                num_params = len(optim.state)
                for param_state in optim.state.values():
                    param_state.clear()
                logger.info(
                    f"[reset_gan_optimizers] Cleared Adam moments for {name} optimizer "
                    f"({num_params} param tensors reset)"
                )
        if train_cfg.get("reset_flow_optimizer", False):
            num_params = len(self.optim_flow.state)
            for param_state in self.optim_flow.state.values():
                param_state.clear()
            logger.info(
                f"[reset_flow_optimizer] Cleared Adam moments for flow optimizer "
                f"({num_params} param tensors reset)"
            )
        # Load scalers (backward compat: old checkpoints may lack scaler_flow)
        if "scaler_flow" in ckpt:
            self.scaler_flow.load_state_dict(ckpt["scaler_flow"])
        if "scaler_gen" in ckpt:
            self.scaler_gen.load_state_dict(ckpt["scaler_gen"])
            self.scaler_disc.load_state_dict(ckpt["scaler_disc"])
        elif "scaler" in ckpt:
            self.scaler_gen.load_state_dict(ckpt["scaler"])
            self.scaler_disc.load_state_dict(ckpt["scaler"])
            logger.info("Migrated single scaler → separate gen/disc scalers")
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        logger.info(f"Resumed from {path} (epoch {self.epoch}, step {self.global_step})")
