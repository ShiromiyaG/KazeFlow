"""
KazeFlow Fine-Tuning Training Pipeline.

Joint training of RFM (Rectified Flow Matching) + ChouwaGAN vocoder + discriminator
on a single-speaker (or small multi-speaker) dataset.

Workflow:
  Stage 1: Pretrain RFM on many speakers (pretrain.py)
  Stage 2: Train vocoder on frozen RFM (vocoder_trainer.py)
  Stage 3: Fine-tune both jointly ÔÇö THIS FILE

Precision options:
- FP32 (default): Standard full precision
- FP16 + AMP: Mixed precision with automatic scaling (faster, less VRAM)
- TF32: TensorFloat-32 on Ampere+ GPUs (faster matmuls, no accuracy loss)
- TF32+BF16: Recommended for Ampere+ (best throughput, no accuracy loss)

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

from kazeflow.models import build_mel_model
from kazeflow.models.vocoder import build_vocoder, EMAGenerator
from kazeflow.models.discriminator import build_discriminator
from kazeflow.train.dataset import create_dataloader
from kazeflow.train.losses import (
    GradientBalancer,
    LeCamEMA,
    discriminator_loss_lsgan,
    discriminator_loss_hinge,
    discriminator_loss_softplus,
    envelope_loss,
    feature_loss,
    generator_loss_lsgan,
    generator_loss_hinge,
    generator_loss_soft_hinge,
    mel_spectrogram_loss,
    mel_spectral_convergence_loss,
    multi_resolution_stft_loss,
    phase_continuity_loss,
    r1_gradient_penalty,
)

logger = logging.getLogger("kazeflow.train")

def _log_section(title: str) -> None:
    logger.info(f"ÔöÇÔöÇ {title} {'ÔöÇ' * max(1, 45 - len(title))}")


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
    Joint fine-tuning trainer for KazeFlow:
    - Rectified Flow Matching (RFM): trains contentÔåÆmel velocity field
    - ChouwaGAN vocoder: trains melÔåÆwaveform with GAN losses
    - Discriminator: adversarial training for vocoder quality

    Load a pretrain checkpoint (from KazeFlowPretrainer) or a vocoder
    joint checkpoint (from KazeFlowVocoderTrainer) to start fine-tuning.

    Precision modes:
    - "fp32": Standard (default)
    - "fp16": FP16 mixed precision with AMP + GradScaler
    - "tf32": TF32 matmuls on Ampere+ (falls back to fp32 if unsupported)
    - "tf32_bf16": TF32 + BF16 AMP (recommended for Ampere+)
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

        # ÔöÇÔöÇ Precision setup ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        _log_section("Precision")
        precision = train_cfg.get("precision", "fp32").lower()
        self.use_amp = False
        self.amp_dtype = torch.float32

        if precision == "tf32_fp16":
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
        elif precision == "fp32_fp16":
            if _check_fp16_support():
                self.use_amp = True
                self.amp_dtype = torch.float16
                logger.info("FP32 + FP16 AMP enabled")
            else:
                logger.warning("FP16 requested but GPU doesn't support it efficiently. Using FP32.")
        elif precision == "fp32":
            logger.info("Using FP32 precision")
        else:
            logger.warning(f"Unknown precision '{precision}', falling back to FP32")

        self._needs_scaler = (self.use_amp and self.amp_dtype == torch.float16)

        # ÔöÇÔöÇ Extra CUDA performance knobs ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")
        if _check_tf32_support():
            torch.set_float32_matmul_precision("high")
            logger.info("float32 matmul precision set to 'high' (TF32, Ampere+)")

        # ÔöÇÔöÇ Models ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        self.architecture = model_cfg.get("architecture", "rfm")
        self.flow = build_mel_model(
            self.architecture, **model_cfg["flow_matching"]
        ).to(self.device)

        vocoder_type = model_cfg.get("vocoder_type", "chouwa_gan")
        self.vocoder = build_vocoder(
            vocoder_type,
            sr=model_cfg["sample_rate"],
            **model_cfg["vocoder"]
        ).to(self.device)

        disc_type = model_cfg.get("discriminator_type", vocoder_type)
        self.discriminator = build_discriminator(
            disc_type,
            sample_rate=model_cfg["sample_rate"],
            **model_cfg["discriminator"]
        ).to(self.device)

        # RFM does not use a mel discriminator
        self.mel_disc = None

        # Speaker embedding
        n_speakers = model_cfg.get("n_speakers", 1)
        spk_dim = model_cfg.get("speaker_embed_dim", 256)
        self.speaker_embed = nn.Embedding(n_speakers, spk_dim).to(self.device)

        # ÔöÇÔöÇ Per-layer gradient clip on vocoder output layer ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        _log_section("Compilation")
        _cp_clip = train_cfg.get("conv_post_grad_clip", 0.0)
        if _cp_clip > 0 and hasattr(self.vocoder, "register_output_grad_clip"):
            self.vocoder.register_output_grad_clip(_cp_clip)
            logger.info(f"Vocoder output layer per-param gradient clip: {_cp_clip}")

        # ÔöÇÔöÇ torch.compile ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        if train_cfg.get("torch_compile", False):
            compile_mode = train_cfg.get("compile_mode", "default")
            compile_disc = train_cfg.get("compile_disc", False)
            try:
                self.flow.estimator = torch.compile(
                    self.flow.estimator, mode=compile_mode)
                if hasattr(self.vocoder, "get_compilable_module"):
                    mod = self.vocoder.get_compilable_module()
                    compiled = torch.compile(mod, mode=compile_mode)
                    for name, child in self.vocoder.named_children():
                        if child is mod:
                            setattr(self.vocoder, name, compiled)
                            break
                else:
                    self.vocoder = torch.compile(self.vocoder, mode=compile_mode)
                if compile_disc:
                    self.discriminator = torch.compile(
                        self.discriminator, mode=compile_mode)
                disc_label = " + discriminator" if compile_disc else "; discriminator skipped"
                logger.info(f"torch.compile (mode='{compile_mode}') applied (flow + vocoder{disc_label})")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")

        # ÔöÇÔöÇ Optimizers ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        lr_flow = train_cfg["learning_rate_flow"]
        lr_voc = train_cfg["learning_rate_vocoder"]
        lr_disc = train_cfg["learning_rate_disc"]
        betas = tuple(train_cfg["betas"])
        betas_flow = tuple(train_cfg.get("betas_flow", [0.95, 0.999]))
        wd = train_cfg.get("weight_decay", 0.01)
        fused = torch.cuda.is_available()

        def _make_adamw(params, lr, betas, **kwargs):
            return torch.optim.AdamW(params, lr=lr, betas=betas, fused=fused, **kwargs)

        flow_decay, flow_no_decay = [], []
        for name, p in list(self.flow.named_parameters()) + \
                        [(f"spk.{n}", p) for n, p in self.speaker_embed.named_parameters()]:
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or "norm" in name or "out_proj" in name:
                flow_no_decay.append(p)
            else:
                flow_decay.append(p)
        self.optim_flow = _make_adamw([
            {"params": flow_decay, "weight_decay": wd},
            {"params": flow_no_decay, "weight_decay": 0.0},
        ], lr=lr_flow, betas=betas_flow)
        betas_vocoder = tuple(train_cfg.get("betas_vocoder", [0.9, 0.999]))
        betas_disc = tuple(train_cfg.get("betas_disc", [0.9, 0.999]))
        self.optim_vocoder = _make_adamw(
            self.vocoder.parameters(),
            lr=lr_voc, betas=betas_vocoder, weight_decay=wd,
        )
        _disc_cfg = model_cfg.get("discriminator", {})
        if _disc_cfg.get("use_spectral_norm", False):
            self.optim_disc = _make_adamw(
                self.discriminator.parameters(),
                lr=lr_disc, betas=betas_disc, weight_decay=0.0,
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
            self.optim_disc = _make_adamw([
                {"params": disc_body, "weight_decay": wd},
                {"params": disc_conv_post, "weight_decay": 0.0},
            ], lr=lr_disc, betas=betas_disc)

        # RFM has no mel discriminator
        self.optim_mel_disc = None

        # ÔöÇÔöÇ Schedulers ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        lr_decay = train_cfg.get("lr_decay", 0.9999)
        _lr_sched_type = train_cfg.get("lr_scheduler", "exponential")
        _eta_min_ratio = train_cfg.get("lr_eta_min_ratio", 0.01)
        _lr_warmup_epochs = train_cfg.get("lr_warmup_epochs", 0)
        _disc_restart_period = train_cfg.get("disc_restart_period", 50)
        _disc_restart_mult = train_cfg.get("disc_restart_mult", 2)

        def _make_sched(optim, t_max, is_disc=False):
            if _lr_sched_type == "cosine":
                _base_lr = optim.param_groups[0]["lr"]
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=max(1, t_max),
                    eta_min=_base_lr * _eta_min_ratio)
            if _lr_sched_type == "cosine_warmup_restarts":
                _base_lr = optim.param_groups[0]["lr"]
                if is_disc:
                    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optim, T_0=max(1, _disc_restart_period),
                        T_mult=_disc_restart_mult,
                        eta_min=_base_lr * _eta_min_ratio)
                if _lr_warmup_epochs > 0 and t_max > _lr_warmup_epochs:
                    _wu = torch.optim.lr_scheduler.LinearLR(
                        optim, start_factor=0.01, end_factor=1.0,
                        total_iters=_lr_warmup_epochs)
                    _cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optim, T_max=max(1, t_max - _lr_warmup_epochs),
                        eta_min=_base_lr * _eta_min_ratio)
                    return torch.optim.lr_scheduler.SequentialLR(
                        optim, schedulers=[_wu, _cos],
                        milestones=[_lr_warmup_epochs])
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=max(1, t_max),
                    eta_min=_base_lr * _eta_min_ratio)
            return torch.optim.lr_scheduler.ExponentialLR(
                optim, gamma=lr_decay)

        _total_epochs = train_cfg["epochs"]
        self.sched_flow = _make_sched(self.optim_flow, _total_epochs)
        self.sched_vocoder = _make_sched(self.optim_vocoder, _total_epochs)
        self.sched_disc = _make_sched(self.optim_disc, _total_epochs, is_disc=True)
        self.sched_mel_disc = None

        if _lr_sched_type == "cosine":
            _sched_info = f" (eta_min_ratio={_eta_min_ratio})"
        elif _lr_sched_type == "cosine_warmup_restarts":
            _sched_info = (f" (warmup={_lr_warmup_epochs}ep, "
                           f"disc_restart={_disc_restart_period}x{_disc_restart_mult}, "
                           f"eta_min_ratio={_eta_min_ratio})")
        else:
            _sched_info = f" (gamma={lr_decay})"
        logger.info(f"LR scheduler: {_lr_sched_type}{_sched_info}")

        # ÔöÇÔöÇ AMP GradScalers ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        self.scaler_flow = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)
        self.scaler_gen = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)
        self.scaler_disc = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)

        # ÔöÇÔöÇ Logging ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"), max_queue=1)
        self.global_step = 0
        self.epoch = 0
        self._eval_count = 0

        # ÔöÇÔöÇ Config shortcuts ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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

        # RFM does not use mel adversarial losses

        # GAN loss type
        _gan_type = train_cfg.get("gan_loss_type", "hinge")
        _use_san = model_cfg.get("discriminator", {}).get("use_san", False)

        if _gan_type in ("hinge", "soft_hinge"):
            if _use_san:
                self._disc_loss_fn = discriminator_loss_softplus
                self._gen_loss_fn = generator_loss_soft_hinge
            else:
                self._disc_loss_fn = discriminator_loss_hinge
                self._gen_loss_fn = (
                    generator_loss_soft_hinge if _gan_type == "soft_hinge"
                    else generator_loss_hinge
                )
        else:
            self._disc_loss_fn = discriminator_loss_lsgan
            self._gen_loss_fn = generator_loss_lsgan

        logger.info(
            f"GAN loss: {_gan_type}"
            + (f" + SAN (softplus disc + soft_hinge gen)"
               if _use_san else "")
        )
        self.grad_clip_flow = train_cfg["grad_clip_flow"]
        self.grad_clip_voc = train_cfg["grad_clip_vocoder"]
        self.grad_clip_disc = train_cfg["grad_clip_disc"]
        self.c_lecam = train_cfg.get("c_lecam", 0.0)
        self.lecam = LeCamEMA(decay=train_cfg.get("lecam_ema_decay", 0.999)) if self.c_lecam > 0 else None
        self.c_env = train_cfg.get("c_env", 0.0)
        self.c_phase = train_cfg.get("c_phase", 0.0)
        self.cfm_grad_accum = max(1, int(train_cfg.get("cfm_grad_accum", 1)))

        # ÔöÇÔöÇ Gradient Balancer ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        self.use_gradient_balancer = train_cfg.get("use_gradient_balancer", False)
        if self.use_gradient_balancer:
            _bal_weights = {
                "gen": self.c_gen,
                "fm": self.c_fm,
                "mel": self.c_mel,
                "mrstft": self.c_mrstft,
            }
            self.gradient_balancer = GradientBalancer(
                weights=_bal_weights,
                ema_decay=train_cfg.get("balancer_ema_decay", 0.999),
            )
            logger.info(
                f"Gradient balancer enabled "
                f"(weights={_bal_weights}, ema_decay={train_cfg.get('balancer_ema_decay', 0.999)})"
            )
        else:
            self.gradient_balancer = None

        self._mel_basis = None
        self._reference = None

        # ÔöÇÔöÇ EMA Generator ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
        ema_decay = train_cfg.get("ema_decay", 0.999)
        self.ema_vocoder = EMAGenerator(self.vocoder, decay=ema_decay)
        self.ema_vocoder.to(self.device)
        self.ema_flow = EMAGenerator(self.flow, decay=ema_decay)
        self.ema_flow.to(self.device)
        logger.info(f"EMA vocoder+flow initialized (decay={ema_decay})")

    def _eval_infer(self, mel_ref, spin_ref, f0_ref, spk_id_ref,
                     ode_steps_override=None):
        """Run the full generation pipeline in eval mode on a reference sample."""
        ema_flow = self.ema_flow.get_model()
        ema_voc = self.ema_vocoder.get_model()
        ema_flow.eval()
        ema_voc.eval()
        with torch.no_grad():
            g = self.speaker_embed(spk_id_ref).unsqueeze(-1)
            B, _, T = mel_ref.shape
            x_mask = torch.ones(B, 1, T, device=self.device)
            f0_expanded = f0_ref.unsqueeze(1)

            infer_cfg = self.config.get("inference", {})
            # RFM: ODE-based mel generation
            ode_steps = ode_steps_override or infer_cfg.get(
                "ode_steps", self.config["train"].get("ode_steps_infer", 4))
            mel_hat = ema_flow.sample(
                content=spin_ref,
                f0=f0_expanded,
                x_mask=x_mask,
                g=g,
                n_steps=ode_steps,
                method=infer_cfg.get("ode_method", "midpoint"),
                guidance_scale=infer_cfg.get("guidance_scale", 1.0),
            )

            wav_hat = ema_voc(mel_hat, f0_ref, g=g)
        return mel_hat, wav_hat, ode_steps

    def _get_reference_sample(self, dataloader):
        """Get a deterministic, full-length reference sample for eval inference."""
        if self._reference is not None:
            return self._reference

        dataset = dataloader.dataset
        best_idx = 0
        best_len = 0
        for i, (wav_path, spk_id) in enumerate(dataset.entries):
            stem = Path(wav_path).stem
            root = Path(dataset.dataset_root)
            mel_path = root / "mel" / f"{stem}.npy"
            if not mel_path.exists():
                continue
            try:
                length = np.load(mel_path, mmap_mode="r").shape[-1]
            except Exception:
                continue
            if length > best_len:
                best_len = length
                best_idx = i

        wav_path, spk_id = dataset.entries[best_idx]
        stem = Path(wav_path).stem
        root = Path(dataset.dataset_root)

        mel = torch.from_numpy(np.load(root / "mel" / f"{stem}.npy")).float()
        spin = torch.from_numpy(np.load(root / "spin" / f"{stem}.npy")).float().T
        f0 = torch.from_numpy(np.load(root / "f0" / f"{stem}.npy")).float()

        if spin.shape[1] < mel.shape[1]:
            spin = torch.nn.functional.interpolate(
                spin.unsqueeze(0), size=mel.shape[1], mode="linear", align_corners=False
            ).squeeze(0)

        min_len = min(mel.shape[1], spin.shape[1], f0.shape[0])
        mel = mel[:, :min_len]
        spin = spin[:, :min_len]
        f0 = f0[:min_len]

        self._reference = (
            mel.unsqueeze(0).to(self.device),
            spin.unsqueeze(0).to(self.device),
            f0.unsqueeze(0).to(self.device),
            torch.tensor([spk_id], dtype=torch.long, device=self.device),
        )
        logger.info(f"Eval reference: {stem} ({min_len} frames)")
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
        """Main training loop."""
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
            content_embedder=self.config["preprocess"].get("content_embedder", "spin_v2"),
        )

        epochs = train_cfg["epochs"]
        save_every = train_cfg["save_every"]
        log_every = train_cfg["log_every"]

        _log_section("Start")
        logger.info(f"Architecture: {self.architecture}")
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Dataset: {len(dataloader.dataset)} samples")

        for epoch in range(self.epoch, epochs):
            if stop_event is not None and stop_event.is_set():
                break

            self.epoch = epoch
            self.flow.train()
            self.vocoder.train()
            self.discriminator.train()

            pbar = tqdm(
                dataloader,
                desc=f"[Train] Epoch {epoch+1}/{epochs}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )
            _accum_cfm = 0
            _gn_flow = torch.tensor(0.0)

            # ODE step schedule: stratified sampling
            import random as _random
            _ode_min_cfg = train_cfg.get("ode_steps_train_min", train_cfg.get("ode_steps_train", 2))
            _ode_max_cfg = train_cfg.get("ode_steps_train_max", train_cfg.get("ode_steps_train", 4))
            _eff_ode_max = _ode_max_cfg
            _n_batches = len(dataloader)
            _step_pool = list(range(_ode_min_cfg, _ode_max_cfg + 1))
            _ode_schedule = (_step_pool * ((_n_batches // len(_step_pool)) + 1))[:_n_batches]
            _random.shuffle(_ode_schedule)

            for batch_idx, batch in enumerate(pbar):
                mel, spin, f0, spk_ids, wav_gt = [x.to(self.device) for x in batch]
                g = self.speaker_embed(spk_ids).unsqueeze(-1)
                B, _, T = mel.shape
                x_mask = torch.ones(B, 1, T, device=self.device)
                f0_expanded = f0.unsqueeze(1)

                # ÔöÇÔöÇ Step 1: RFM loss ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
                if _accum_cfm == 0:
                    self.optim_flow.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    loss_cfm = self.flow(
                        x_1=mel, x_mask=x_mask,
                        content=spin, f0=f0_expanded, g=g,
                    )
                if not torch.isfinite(loss_cfm):
                    self.optim_flow.zero_grad(set_to_none=True)
                    self.global_step += 1
                    continue
                self.scaler_flow.scale(loss_cfm / self.cfm_grad_accum).backward()
                _accum_cfm += 1
                _do_flow_step = (_accum_cfm == self.cfm_grad_accum)
                if _do_flow_step:
                    self.scaler_flow.unscale_(self.optim_flow)
                    _gn_flow = torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.grad_clip_flow)
                    self.scaler_flow.step(self.optim_flow)
                    self.ema_flow.update()
                    _accum_cfm = 0

                # ÔöÇÔöÇ Step 2: Generate mel from RFM (for vocoder) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
                _train_cfg = self.config["train"]
                _ode_n = _ode_schedule[batch_idx % len(_ode_schedule)]
                self.flow.eval()
                with torch.no_grad():
                    mel_hat = self.flow.sample(
                        content=spin, f0=f0_expanded,
                        x_mask=x_mask, g=g,
                        n_steps=_ode_n,
                        method=_train_cfg.get("ode_method_train", "midpoint"),
                    )
                self.flow.train()
                _gt_mel_ratio = _train_cfg.get("gt_mel_ratio", 0.0)
                _use_gt_mel = _gt_mel_ratio > 0 and _random.random() < _gt_mel_ratio
                _voc_input_mel = mel.detach() if _use_gt_mel else mel_hat.detach()

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    wav_hat = self.vocoder(_voc_input_mel, f0, g=g.detach())
                    target_len = min(wav_hat.shape[-1], wav_gt.shape[-1])
                    wav_hat = wav_hat[..., :target_len]
                    wav_real_detached = wav_gt[..., :target_len].detach()

                with torch.no_grad():
                    _hat_peak = wav_hat.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
                    _gt_peak = wav_real_detached.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
                    _level_scale = (_hat_peak / _gt_peak).clamp(0.5, 1.0)
                    wav_real_detached = wav_real_detached * _level_scale

                # ÔöÇÔöÇ Discriminator step ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
                gn_disc = None
                d_real_mean = d_fake_mean = 0.0
                loss_disc = torch.tensor(0.0, device=self.device)
                for _disc_i in range(self.n_disc_steps):
                    self.optim_disc.zero_grad()
                    with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                        y_d_rs, y_d_gs, _, _ = self.discriminator(wav_real_detached, wav_hat.detach(), compute_fmaps=False)
                        loss_disc, _d_real, _d_fake = self._disc_loss_fn(y_d_rs, y_d_gs)
                    self.scaler_disc.scale(loss_disc).backward()
                    self.scaler_disc.unscale_(self.optim_disc)
                    gn_disc = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_disc)
                    self.scaler_disc.step(self.optim_disc)
                    self.scaler_disc.update()
                    d_real_mean, d_fake_mean = _d_real, _d_fake

                # ÔöÇÔöÇ Vocoder step ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
                self.optim_vocoder.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(wav_real_detached, wav_hat, compute_fmaps=True)
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
                loss_voc_total = (self.c_gen * loss_gen + self.c_fm * loss_fm + self.c_mel * loss_mel)
                self.scaler_gen.scale(loss_voc_total).backward()
                self.scaler_gen.unscale_(self.optim_vocoder)
                gn_voc = torch.nn.utils.clip_grad_norm_(self.vocoder.parameters(), self.grad_clip_voc)
                self.scaler_gen.step(self.optim_vocoder)
                self.ema_vocoder.update()

                if _do_flow_step:
                    self.scaler_flow.update()
                self.scaler_gen.update()
                self.global_step += 1

                pbar.set_postfix(
                    rfm=f"{loss_cfm.item():.4f}",
                    mel=f"{loss_mel.item():.4f}",
                    gen=f"{loss_gen.item():.4f}",
                    disc=f"{loss_disc.item():.4f}",
                )

                if self.global_step % log_every == 0:
                    step = self.global_step
        if "scaler_flow" in ckpt:
            self.scaler_flow.load_state_dict(ckpt["scaler_flow"])
        if "scaler_gen" in ckpt:
            self.scaler_gen.load_state_dict(ckpt["scaler_gen"])
            self.scaler_disc.load_state_dict(ckpt["scaler_disc"])
        elif "scaler" in ckpt:
            self.scaler_gen.load_state_dict(ckpt["scaler"])
            self.scaler_disc.load_state_dict(ckpt["scaler"])
            logger.info("Migrated single scaler ÔåÆ separate gen/disc scalers")

        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self._eval_count = ckpt.get("eval_count", 0)
        # If transitioning scheduler types, rebuild schedulers positioned at the
        # current epoch so the LR curve continues smoothly from where training left off.
        if _sched_type_now != _sched_type_ckpt and _sched_type_now in ("cosine", "cosine_warmup_restarts"):
            _tc = self.config["train"]
            _T = _tc["epochs"]
            _er = _tc.get("lr_eta_min_ratio", 0.01)
            _wu = _tc.get("lr_warmup_epochs", 0)
            _rp = _tc.get("disc_restart_period", 50)
            _rm = _tc.get("disc_restart_mult", 2)

            def _cosine_at(optim, t_max, at_ep):
                _base = optim.param_groups[0]["lr"]
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=max(1, t_max),
                    eta_min=_base * _er,
                    last_epoch=max(0, at_ep - 1),
                )

            if _sched_type_now == "cosine":
                self.sched_flow    = _cosine_at(self.optim_flow,    _T, self.epoch)
                self.sched_vocoder = _cosine_at(self.optim_vocoder, _T, self.epoch)
                self.sched_disc    = _cosine_at(self.optim_disc,    _T, self.epoch)
                logger.info(
                    f"Cosine schedulers rebuilt at epoch {self.epoch} "
                    f"(T_max={_T})"
                )
            else:  # cosine_warmup_restarts
                _flow_ep = max(0, self.epoch - _wu)
                self.sched_flow    = _cosine_at(self.optim_flow,    max(1, _T - _wu), _flow_ep)
                self.sched_vocoder = _cosine_at(self.optim_vocoder, max(1, _T - _wu), _flow_ep)
                _disc_base = self.optim_disc.param_groups[0]["lr"]
                self.sched_disc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optim_disc, T_0=max(1, _rp), T_mult=_rm,
                    eta_min=_disc_base * _er, last_epoch=self.epoch)
                logger.info(
                    f"cosine_warmup_restarts schedulers rebuilt at epoch {self.epoch} "
                    f"(T={_T}, warmup={_wu}, disc_restart={_rp}x{_rm})"
                )
        logger.info(f"Resumed from {path} (epoch {self.epoch}, step {self.global_step})")
