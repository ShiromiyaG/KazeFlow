"""
KazeFlow Pretraining Pipeline.

Multi-speaker pretraining to create a base model that can be
fine-tuned for single-speaker voice conversion.

Differences from fine-tuning (trainer.py):
- Multi-speaker dataset with many speakers (n_speakers >> 1)
- Longer training with lower learning rates
- No FAISS index (learned during fine-tuning)
- Saves pretrain checkpoints that can be loaded for fine-tuning

Usage:
    python -m kazeflow.train.pretrain \\
        --config kazeflow/configs/pretrain.json \\
        --filelist dataset/filelist.txt \\
        --dataset-root dataset/ \\
        --output-dir logs/pretrain/
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
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from kazeflow.models.flow_matching import ConditionalFlowMatching
from kazeflow.models.vocoder import build_vocoder, EMAGenerator
from kazeflow.models.discriminator import build_discriminator
from kazeflow.train.dataset import create_dataloader
from kazeflow.train.losses import (
    LeCamEMA,
    discriminator_loss_lsgan,
    discriminator_loss_hinge,
    discriminator_loss_softplus,
    feature_loss,
    generator_loss_lsgan,
    generator_loss_hinge,
    generator_loss_soft_hinge,
    mel_spectrogram_loss,
    multi_resolution_stft_loss,
    r1_gradient_penalty,
)

logger = logging.getLogger("kazeflow.pretrain")

def _log_section(title: str) -> None:
    logger.info(f"── {title} {'─' * max(1, 45 - len(title))}")


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


class KazeFlowPretrainer:
    """
    Multi-speaker pretraining for KazeFlow.

    Creates a base model trained on diverse speakers that generalizes
    well to unseen speakers when fine-tuned.

    Key differences from KazeFlowTrainer:
    - Supports multi-GPU via DDP
    - Phase-based training: CFM warmup → joint training
    - Saves pretrain checkpoint format compatible with fine-tuning
    """

    def __init__(
        self,
        config: dict,
        output_dir: str,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        model_cfg = config["model"]
        train_cfg = config["train"]

        if self.is_main:
            with open(self.output_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # ── Precision ────────────────────────────────────────────────────
        _log_section("Precision")
        precision = train_cfg.get("precision", "fp32").lower()
        self.use_amp = False
        self.amp_dtype = torch.float32

        def _has_tf32():
            return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        def _enable_tf32():
            if _has_tf32():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                return True
            return False

        if precision == "tf32_bf16":
            tf32_ok = _enable_tf32()
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
            if _enable_tf32():
                logger.info("TF32 enabled (Ampere+ GPU detected)")
            else:
                logger.warning("TF32 requested but GPU doesn't support it (need sm_80+). Using FP32.")
        elif precision == "fp32_fp16":
            self.use_amp = True
            self.amp_dtype = torch.float16
            logger.info("FP32 + FP16 AMP enabled")
        elif precision == "tf32_fp16":
            _enable_tf32()
            self.use_amp = True
            self.amp_dtype = torch.float16
            logger.info("TF32 + FP16 AMP enabled")
        elif precision == "fp32":
            logger.info("Using FP32 precision")
        else:
            logger.warning(f"Unknown precision '{precision}', falling back to FP32")

        # GradScaler is only needed for FP16 (BF16 has full FP32 dynamic range)
        self._needs_scaler = (self.use_amp and self.amp_dtype == torch.float16)

        # ── Extra CUDA performance knobs ────────────────────────────────
        # cudnn.benchmark: auto-tunes cuDNN conv algorithms for fixed input
        # sizes — free speed-up since segment_frames is always constant.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")
        # float32_matmul_precision: tells PyTorch to prefer TF32 for all
        # float32 matmuls (complements allow_tf32 for Ampere+ GPUs).
        if _has_tf32():
            torch.set_float32_matmul_precision("high")
            logger.info("float32 matmul precision set to 'high' (TF32, Ampere+)")

        # ── Models ───────────────────────────────────────────────────────
        self.flow = ConditionalFlowMatching(
            **model_cfg["flow_matching"]
        ).to(self.device)

        vocoder_type = model_cfg.get("vocoder_type", "chouwa_gan")
        self.vocoder = build_vocoder(
            vocoder_type,
            sr=model_cfg["sample_rate"],
            **model_cfg["vocoder"],
        ).to(self.device)

        disc_type = model_cfg.get("discriminator_type", vocoder_type)
        self.discriminator = build_discriminator(
            disc_type,
            sample_rate=model_cfg["sample_rate"],
            **model_cfg["discriminator"],
        ).to(self.device)

        n_speakers = model_cfg.get("n_speakers", 1)
        spk_dim = model_cfg.get("speaker_embed_dim", 256)
        self.speaker_embed = nn.Embedding(n_speakers, spk_dim).to(self.device)

        # ── Per-layer gradient clip on vocoder output layer ────────────
        _log_section("Compilation")
        # The vocoder output layer receives the bulk of adversarial gradient
        # norm. Per-layer clipping prevents it from dominating the global norm
        # and starving deeper layers of gradient signal.
        _cp_clip = train_cfg.get("conv_post_grad_clip", 0.0)
        if _cp_clip > 0 and hasattr(self.vocoder, "register_output_grad_clip"):
            self.vocoder.register_output_grad_clip(_cp_clip)
            logger.info(f"Vocoder output layer per-param gradient clip: {_cp_clip}")

        # ── torch.compile ────────────────────────────────────────────────
        if train_cfg.get("torch_compile", False):
            compile_mode = train_cfg.get("compile_mode", "default")
            compile_disc = train_cfg.get("compile_disc", False)
            try:
                self.flow.estimator = torch.compile(
                    self.flow.estimator, mode=compile_mode)
                if hasattr(self.vocoder, "get_compilable_module"):
                    mod = self.vocoder.get_compilable_module()
                    compiled = torch.compile(mod, mode=compile_mode)
                    # Replace the attribute on vocoder that points to this module
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
                logger.warning(f"torch.compile failed: {e}")

        # ── DDP wrapping ─────────────────────────────────────────────────
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.flow = DDP(self.flow, device_ids=[rank])
            self.vocoder = DDP(self.vocoder, device_ids=[rank])
            self.discriminator = DDP(self.discriminator, device_ids=[rank])
            self.speaker_embed = DDP(self.speaker_embed, device_ids=[rank])

        # ── Optimizers ───────────────────────────────────────────────────
        lr_flow = train_cfg["learning_rate_flow"]
        lr_voc = train_cfg["learning_rate_vocoder"]
        lr_disc = train_cfg["learning_rate_disc"]
        betas = tuple(train_cfg["betas"])
        # CFM is non-adversarial — higher β₁ smooths the noisy gradients
        # from double stochasticity (random t + random x₀).  GAN betas
        # (0.8, 0.99) are too responsive for this static loss landscape.
        betas_flow = tuple(train_cfg.get("betas_flow", [0.95, 0.999]))
        wd = train_cfg.get("weight_decay", 0.01)
        fused = torch.cuda.is_available()

        def _make_adamw(params, lr, betas, **kwargs):
            return torch.optim.AdamW(params, lr=lr, betas=betas, fused=fused, **kwargs)

        # Split flow parameters: exclude biases, norms, and zero-init
        # out_proj from weight decay.  WD on out_proj counteracts the
        # zero-init by pulling weights back toward zero, creating a
        # premature equilibrium that stalls CFM loss.
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
        # Higher β₁ for vocoder/disc smooths the reactive gradient cycle
        # that amplifies gn_voc as the model learns higher-amplitude audio.
        betas_vocoder = tuple(train_cfg.get("betas_vocoder", [0.9, 0.999]))
        betas_disc = tuple(train_cfg.get("betas_disc", [0.9, 0.999]))
        self.optim_vocoder = _make_adamw(
            self.vocoder.parameters(),
            lr=lr_voc, betas=betas_vocoder, weight_decay=wd,
        )
        # Selective weight decay for discriminator:
        # - Spectral norm: wd=0 everywhere (SN renormalizes, WD is wasted)
        # - SAN: wd>0 for body (weight_norm magnitude `g` drifts without it),
        #        wd=0 for conv_post (L2-normed by SAN, WD fights parametrization)
        # - Otherwise: wd=wd everywhere
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

        # ── Schedulers ───────────────────────────────────────────────────
        _log_section("Training")
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
        _cfm_wu = train_cfg.get("cfm_warmup_epochs", 50)
        self.sched_flow = _make_sched(self.optim_flow, _total_epochs)
        self.sched_vocoder = _make_sched(
            self.optim_vocoder, _total_epochs - _cfm_wu)
        self.sched_disc = _make_sched(
            self.optim_disc, _total_epochs - _cfm_wu, is_disc=True)
        if _lr_sched_type == "cosine":
            _sched_info = f" (eta_min_ratio={_eta_min_ratio})"
        elif _lr_sched_type == "cosine_warmup_restarts":
            _sched_info = (f" (warmup={_lr_warmup_epochs}ep, "
                           f"disc_restart={_disc_restart_period}x{_disc_restart_mult}, "
                           f"eta_min_ratio={_eta_min_ratio})")
        else:
            _sched_info = f" (gamma={lr_decay})"
        logger.info(f"LR scheduler: {_lr_sched_type}{_sched_info}")

        # ── AMP GradScalers (only active with FP16) ────────────────────
        # CRITICAL: Separate scalers for generator path vs discriminator.
        # A single shared scaler causes cross-contamination: when the disc
        # hits inf gradients the scale reduction affects the generator too,
        # and vice versa. This interaction is the primary cause of NaN
        # after ~100k steps in adversarial training with FP16.
        self.scaler_flow = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)
        self.scaler_gen = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)
        self.scaler_disc = torch.amp.GradScaler('cuda', enabled=self._needs_scaler)

        if self.is_main:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"), max_queue=1)
        self.global_step = 0
        self.epoch = 0
        self._eval_count = 0  # total evals run; used to bucket audio into groups of 10

        # Config shortcuts
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
        _use_san = model_cfg["discriminator"].get("use_san", False)

        if _gan_type in ("hinge", "soft_hinge"):
            if _use_san:
                # SAN's L2-normalized conv_post caps score magnitude at ~±0.5,
                # which can never satisfy the hinge margin of 1.0 — the disc
                # would receive permanent maximum gradient.  Softplus (logistic)
                # has no hard margin; gradient decays smoothly as confidence
                # grows, matching SAN's bounded range naturally.
                self._disc_loss_fn = discriminator_loss_softplus
                self._gen_loss_fn = generator_loss_soft_hinge
            else:
                self._disc_loss_fn = discriminator_loss_hinge
                if _gan_type == "soft_hinge":
                    self._gen_loss_fn = generator_loss_soft_hinge
                else:
                    self._gen_loss_fn = generator_loss_hinge
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
        self.cfm_grad_accum = max(1, int(train_cfg.get("cfm_grad_accum", 1)))

        self._mel_basis = None

        # Reference sample for eval inference (lazy-initialized)
        self._reference = None

        # ── EMA Generator ────────────────────────────────────────────────
        ema_decay = train_cfg.get("ema_decay", 0.999)
        # EMA wraps the unwrapped model (before DDP) for clean state_dict
        self.ema_vocoder = EMAGenerator(
            self._unwrap_model(self.vocoder), decay=ema_decay,
        )
        self.ema_vocoder.to(self.device)
        self.ema_flow = EMAGenerator(
            self._unwrap_model(self.flow), decay=ema_decay,
        )
        self.ema_flow.to(self.device)
        logger.info(f"EMA vocoder+flow initialized (decay={ema_decay})")



    def _get_mel_basis(self):
        if self._mel_basis is None:
            import torchaudio
            self._mel_basis = torchaudio.functional.melscale_fbanks(
                n_freqs=self.config["model"]["n_fft"] // 2 + 1,
                f_min=0.0, f_max=self.sample_rate / 2.0,
                n_mels=self.n_mels, sample_rate=self.sample_rate,
            ).T.to(self.device)
        return self._mel_basis

    def _eval_infer(self, mel_ref, spin_ref, f0_ref, spk_id_ref):
        """
        Run the full generation pipeline in eval mode on a reference sample.
        Uses EMA vocoder for higher-quality inference.

        Returns:
            mel_hat: (1, n_mels, T) generated mel-spectrogram
            wav_hat: (1, 1, T_audio) generated waveform
        """
        speaker_embed = self._unwrap_model(self.speaker_embed)
        ema_flow = self.ema_flow.get_model()
        ema_voc = self.ema_vocoder.get_model()

        ema_flow.eval()
        ema_voc.eval()
        with torch.no_grad():
            g = speaker_embed(spk_id_ref).unsqueeze(-1)
            B, _, T = mel_ref.shape
            x_mask = torch.ones(B, 1, T, device=self.device)
            f0_expanded = f0_ref.unsqueeze(1)

            infer_cfg = self.config.get("inference", {})
            mel_hat = ema_flow.sample(
                content=spin_ref,
                f0=f0_expanded,
                x_mask=x_mask,
                g=g,
                n_steps=infer_cfg.get("ode_steps", self.config["train"].get("ode_steps_infer", 4)),
                method=infer_cfg.get("ode_method", "midpoint"),
            )

            wav_hat = ema_voc(mel_hat, f0_ref, g=g)
        return mel_hat, wav_hat

    def _get_reference_sample(self, dataloader):
        """
        Get a deterministic, full-length reference sample for eval inference.

        Instead of taking a random segment from the shuffled dataloader (which
        gives inconsistent ~0.5s snippets across restarts), load a full
        utterance directly from disk.  Picks the longest non-mute entry to
        maximize perceptual eval quality.
        """
        if self._reference is not None:
            return self._reference

        dataset = dataloader.dataset
        best_idx = 0
        best_len = 0
        for i, (wav_path, spk_id) in enumerate(dataset.entries):
            stem = Path(wav_path).stem
            if stem == "mute":
                continue
            mel_path = Path(dataset.dataset_root) / "mel" / f"{stem}.npy"
            if mel_path.exists():
                # Read only shape (fast — numpy memmap header)
                mel = np.load(mel_path, mmap_mode="r")
                if mel.shape[1] > best_len:
                    best_len = mel.shape[1]
                    best_idx = i

        wav_path, spk_id = dataset.entries[best_idx]
        stem = Path(wav_path).stem
        root = Path(dataset.dataset_root)

        mel = torch.from_numpy(np.load(root / "mel" / f"{stem}.npy")).float()
        spin = torch.from_numpy(np.load(root / "spin" / f"{stem}.npy")).float().T
        f0 = torch.from_numpy(np.load(root / "f0" / f"{stem}.npy")).float()

        # Interpolate SPIN to mel frame rate (50Hz → 100Hz)
        if spin.shape[1] < mel.shape[1]:
            spin = torch.nn.functional.interpolate(
                spin.unsqueeze(0), size=mel.shape[1], mode="linear",
                align_corners=False,
            ).squeeze(0)

        # Align lengths
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
        logger.info(f"Eval reference: {stem} ({min_len} frames, {min_len * self.hop_length / self.sample_rate:.1f}s)")
        return self._reference

    def _unwrap_model(self, model):
        """Unwrap DDP and torch.compile wrappers to get the raw module."""
        if hasattr(model, "module"):        # DistributedDataParallel
            model = model.module
        if hasattr(model, "_orig_mod"):     # torch.compile (OptimizedModule)
            model = model._orig_mod
        return model

    def train(
        self,
        filelist_path: str,
        dataset_root: str,
        resume_path: str = None,
        stop_event=None,
    ):
        """
        Main pretraining loop.

        Phase 1 (cfm_warmup_epochs): Train only CFM (flow matching)
        Phase 2 (remaining epochs): Train CFM + vocoder + discriminator jointly

        Args:
            stop_event: Optional ``threading.Event``. When set, the loop exits
                cleanly after the current epoch completes and saves a checkpoint.
        """
        train_cfg = self.config["train"]
        cfm_warmup = train_cfg.get("cfm_warmup_epochs", 50)
        epochs = train_cfg["epochs"]
        save_every = train_cfg["save_every"]
        log_every = train_cfg["log_every"]

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
            skip_wav=self.epoch < cfm_warmup,
            content_embedder=self.config["preprocess"].get("content_embedder", "spin_v2"),
        )
        _dl_has_wav = not (self.epoch < cfm_warmup)

        if self.is_main:
            _log_section("Start")
            logger.info(f"Vocoder: {self.config['model'].get('vocoder_type', 'chouwa_gan')}")
            logger.info(f"Pretraining for {epochs} epochs ({cfm_warmup} CFM warmup)")
            logger.info(f"Dataset: {len(dataloader.dataset)} samples")


        for epoch in range(self.epoch, epochs):
            # Check stop signal before starting a new epoch
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop signal received — exiting training loop.")
                break

            self.epoch = epoch
            in_warmup = epoch < cfm_warmup
            if in_warmup:
                phase = "Warmup"
            else:
                phase = "Joint"

            # Recreate dataloader at warmup→joint transition to start
            # loading GT waveforms (skip_wav=False) for vocoder training.
            if not in_warmup and not _dl_has_wav:
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
                _dl_has_wav = True
                if self.is_main:
                    logger.info("Warmup complete — dataloader now loads GT audio")

            self.flow.train()
            self.vocoder.train()
            self.discriminator.train()

            pbar = tqdm(
                dataloader,
                desc=f"[{phase}] Epoch {epoch+1}/{epochs}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
                disable=not self.is_main,
            )
            _accum_cfm = 0
            _gn_flow = torch.tensor(0.0)

            # Progressive ODE: precompute effective max for this epoch
            if not in_warmup:
                import math as _math
                import random as _random
                _ode_min_cfg = train_cfg.get("ode_steps_train_min", train_cfg.get("ode_steps_train", 1))
                _ode_max_cfg = train_cfg.get("ode_steps_train_max", train_cfg.get("ode_steps_train", 1))
                if train_cfg.get("progressive_ode", False) and _ode_min_cfg < _ode_max_cfg:
                    _ramp = train_cfg.get("ode_ramp_epochs", epochs - cfm_warmup)
                    _joint_ep = epoch - cfm_warmup
                    _progress = min(1.0, max(0.0, _joint_ep / max(1, _ramp)))
                    _eff_ode_max = max(_ode_min_cfg, int(round(
                        _ode_min_cfg + (_ode_max_cfg - _ode_min_cfg) * _progress)))
                else:
                    _eff_ode_max = _ode_max_cfg

            for batch_idx, batch in enumerate(pbar):
                mel, spin, f0, spk_ids, wav_gt = [x.to(self.device) for x in batch]
                g = self.speaker_embed(spk_ids).unsqueeze(-1)
                B, _, T = mel.shape
                x_mask = torch.ones(B, 1, T, device=self.device)
                f0_expanded = f0.unsqueeze(1)

                # ── CFM Loss ─────────────────────────────────────────
                # Gradient accumulation: zero_grad only at start of window
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
                    if stop_event is not None and stop_event.is_set():
                        break
                    continue

                # Scale loss to average gradients over accumulation window
                # During warmup, always step every batch (no accumulation needed)
                effective_accum = 1 if in_warmup else self.cfm_grad_accum
                self.scaler_flow.scale(loss_cfm / effective_accum).backward()
                _accum_cfm += 1
                _do_flow_step = (_accum_cfm == effective_accum)

                if _do_flow_step:
                    self.scaler_flow.unscale_(self.optim_flow)
                    _gn_flow = torch.nn.utils.clip_grad_norm_(
                        self.flow.parameters(), self.grad_clip_flow)
                    self.scaler_flow.step(self.optim_flow)
                    self.ema_flow.update()
                    _accum_cfm = 0

                if in_warmup:
                    if _do_flow_step:
                        self.scaler_flow.update()
                    # Phase 1: Only CFM, skip vocoder/discriminator
                    self.global_step += 1

                    pbar.set_postfix(cfm=f"{loss_cfm.item():.4f}")
                    if self.is_main and self.global_step % log_every == 0:
                        step = self.global_step
                        self.writer.add_scalar(
                            "loss/cfm", loss_cfm.item(), step)
                        self.writer.add_scalar(
                            "grad_norm/flow", _gn_flow.item(), step)
                        self.writer.add_scalar(
                            "lr/flow", self.optim_flow.param_groups[0]["lr"], step)

                    if stop_event is not None and stop_event.is_set():
                        break
                    continue

                # ── Phase 2: Joint training ──────────────────────────
                # ODE step count — log-uniform within [ode_min, eff_max]
                # (eff_max ramps up with progressive_ode, else equals ode_max)
                if _ode_min_cfg >= _eff_ode_max:
                    _ode_n = _ode_min_cfg
                else:
                    _ode_n = int(round(_math.exp(
                        _random.uniform(_math.log(_ode_min_cfg), _math.log(_eff_ode_max))
                    )))
                self.flow.eval()
                with torch.no_grad():
                    mel_hat = self.flow.sample(
                        content=spin, f0=f0_expanded,
                        x_mask=x_mask, g=g,
                        n_steps=_ode_n,
                        method=train_cfg.get("ode_method_train", "euler"),
                    )
                self.flow.train()

                # GT mel mixing: with probability gt_mel_ratio, feed the
                # vocoder the ground-truth mel instead of the CFM prediction.
                # This gives the vocoder clean targets to learn faithful
                # waveform reconstruction, breaking the cascaded-error plateau.
                _gt_mel_ratio = train_cfg.get("gt_mel_ratio", 0.0)
                _use_gt_mel = _gt_mel_ratio > 0 and _random.random() < _gt_mel_ratio
                _voc_input_mel = mel.detach() if _use_gt_mel else mel_hat.detach()

                # Vocoder forward — only on generated mel (single forward)
                # wav_gt from dataset provides the real audio directly,
                # avoiding a second vocoder forward that would double VRAM.
                # g.detach(): speaker_embed graph was freed by CFM backward;
                # vocoder loss should not update speaker_embed (it's in optim_flow).
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    wav_hat = self.vocoder(_voc_input_mel, f0, g=g.detach())

                    # Align lengths (vocoder iSTFT output may differ slightly)
                    target_len = min(wav_hat.shape[-1], wav_gt.shape[-1])
                    wav_hat = wav_hat[..., :target_len]
                    wav_real_det = wav_gt[..., :target_len].detach()

                # Discriminator — repeated n_disc_steps times per gen step
                gn_disc = None
                d_real = d_fake = 0.0
                loss_disc = torch.tensor(0.0, device=self.device)
                loss_lecam = torch.tensor(0.0, device=self.device)
                _apply_r1 = False
                for _disc_i in range(self.n_disc_steps if not in_warmup else 0):
                    self.optim_disc.zero_grad()
                    # R1 gradient penalty needs grad w.r.t. real audio
                    _apply_r1 = (self.c_r1 > 0
                                 and self.global_step % self.r1_interval == 0)
                    if _apply_r1:
                        wav_real_det.requires_grad_(True)
                    with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                        y_d_rs, y_d_gs, _, _ = self.discriminator(
                            wav_real_det, wav_hat.detach(), compute_fmaps=False)
                        loss_disc, _d_real, _d_fake = self._disc_loss_fn(y_d_rs, y_d_gs)

                    # LeCam regularization (prevents overconfident disc)
                    loss_lecam = torch.tensor(0.0, device=self.device)
                    if self.lecam is not None:
                        loss_lecam = self.lecam.penalty(y_d_rs, y_d_gs)
                        self.lecam.update(y_d_rs, y_d_gs)
                        loss_disc = loss_disc + self.c_lecam * loss_lecam

                    # R1 penalty (FP32, outside autocast)
                    if _apply_r1:
                        r1_pen = r1_gradient_penalty(y_d_rs, wav_real_det)
                        # Scale by interval for lazy regularization (StyleGAN2)
                        loss_disc = loss_disc + (self.c_r1 / 2) * self.r1_interval * r1_pen
                        wav_real_det.requires_grad_(False)

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
                    d_real, d_fake = _d_real, _d_fake

                # Generator (vocoder) losses
                self.optim_vocoder.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    if not in_warmup:
                        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(
                            wav_real_det, wav_hat, compute_fmaps=True)
                        loss_gen = self._gen_loss_fn(y_d_gs)
                        loss_fm = feature_loss(fmap_rs, fmap_gs)
                    else:
                        loss_gen = torch.tensor(0.0, device=self.device)
                        loss_fm = torch.tensor(0.0, device=self.device)
                    loss_mel = mel_spectrogram_loss(
                        wav_real_det, wav_hat,
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
                        wav_real_det, wav_hat)

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
                    loss_voc = (
                        self.c_gen * loss_gen
                        + self.c_fm * loss_fm
                        + self.c_mel * loss_mel
                        + self.c_mrstft * loss_mrstft
                    )
                    self.scaler_gen.scale(loss_voc).backward()
                    self.scaler_gen.unscale_(self.optim_vocoder)
                    gn_voc = torch.nn.utils.clip_grad_norm_(
                        self.vocoder.parameters(), self.grad_clip_voc)
                    self.scaler_gen.step(self.optim_vocoder)
                    self.ema_vocoder.update()

                if _do_flow_step:
                    self.scaler_flow.update()
                self.scaler_gen.update()
                self.global_step += 1

                pbar.set_postfix(
                    cfm=f"{loss_cfm.item():.4f}",
                    mel=f"{loss_mel.item():.4f}",
                    disc=f"{loss_disc.item():.4f}",
                )
                if self.is_main and self.global_step % log_every == 0:
                    step = self.global_step
                    # Losses
                    self.writer.add_scalar("loss/cfm", loss_cfm.item(), step)
                    self.writer.add_scalar("loss/disc", loss_disc.item(), step)
                    self.writer.add_scalar("loss/gen", loss_gen.item(), step)
                    self.writer.add_scalar("loss/fm", loss_fm.item(), step)
                    self.writer.add_scalar("loss/mel", loss_mel.item(), step)
                    if self.c_mrstft > 0:
                        self.writer.add_scalar("loss/mrstft", loss_mrstft.item(), step)
                    if gn_voc is not None:  # loss_voc only exists when step succeeded
                        self.writer.add_scalar("loss/voc_total", loss_voc.item(), step)
                    # Discriminator health
                    if gn_disc is not None:  # d_real/d_fake valid when disc step succeeded
                        self.writer.add_scalar("d_score/real", d_real, step)
                        self.writer.add_scalar("d_score/fake", d_fake, step)
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

            # End of epoch
            # Propagate inner stop-event break to the epoch loop
            if stop_event is not None and stop_event.is_set():
                break

            self.sched_flow.step()
            if not in_warmup:
                self.sched_vocoder.step()
                self.sched_disc.step()

            if self.is_main and (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch + 1)

                # ── Eval inference → TensorBoard ─────────────────────
                # Skip during warmup (vocoder not trained yet)
                if not in_warmup:
                    try:
                        ref_mel, ref_spin, ref_f0, ref_spk = \
                            self._get_reference_sample(dataloader)
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

                        # Log generated audio — grouped 10 evals per TensorBoard panel
                        self._eval_count += 1
                        _audio_group = (self._eval_count - 1) // 10 + 1
                        _audio_tag = f"eval/audio_g{_audio_group:03d}"
                        self.writer.add_audio(
                            _audio_tag,
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

        if self.is_main:
            stopped_early = stop_event is not None and stop_event.is_set()
            if stopped_early:
                logger.info(
                    "Pretraining stopped by user at epoch %d. No checkpoint saved.",
                    self.epoch,
                )
            else:
                self._save_checkpoint(epochs)
                logger.info("Pretraining complete.")
        self._cleanup()

    def _cleanup(self):
        """Close TensorBoard writer and release GPU memory."""
        if self.is_main:
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

    def _save_checkpoint(self, epoch: int):
        """Save pretrain checkpoint (compatible with fine-tuning loader)."""
        path = self.output_dir / f"pretrain_{epoch}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "flow": self._normalize_checkpoint_sd(self._unwrap_model(self.flow).state_dict()),
            "vocoder": self._normalize_checkpoint_sd(self._unwrap_model(self.vocoder).state_dict()),
            "vocoder_ema": self._normalize_checkpoint_sd(self.ema_vocoder.state_dict()),
            "flow_ema": self._normalize_checkpoint_sd(self.ema_flow.state_dict()),
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
            "lr_scheduler": self.config["train"].get("lr_scheduler", "exponential"),
            "eval_count": self._eval_count,
            "is_pretrain": True,
        }, tmp_path)
        tmp_path.replace(path)
        logger.info(f"Saved pretrain checkpoint: {path}")

    @staticmethod
    def _normalize_checkpoint_sd(state_dict: dict) -> dict:
        """
        Normalize a model state dict for loading.  Handles two backward-compat issues:

        1. Checkpoints saved while torch.compile was active store keys with an
           ``_orig_mod.`` prefix (because the old _unwrap_model didn't strip the
           OptimizedModule wrapper).  Strip that prefix so the raw module can
           accept them.

        2. Config-derived buffers (e.g. ``_shared_window``) are computed
           deterministically from hyper-parameters at __init__ time and must
           never be restored from a checkpoint — their shape changes whenever
           n_fft changes.  Drop them so the model keeps its correctly-sized buffer.
        """
        # 1. Strip _orig_mod. prefix (entire model compiled)
        if state_dict and all(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        # Strip _orig_mod. as a sub-module infix (individual sub-module compiled)
        state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}

        # 2. Drop config-derived buffers
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
        if "flow_ema" in ckpt:
            self.ema_flow.load_state_dict(
                self._align_state_dict(self._normalize_checkpoint_sd(ckpt["flow_ema"]), self.ema_flow))
        else:
            logger.warning("No flow EMA state in checkpoint — EMA initialized from flow weights")
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
        _sched_type_now = self.config["train"].get("lr_scheduler", "exponential")
        _sched_type_ckpt = ckpt.get("lr_scheduler", "exponential")
        if _sched_type_now != _sched_type_ckpt:
            logger.warning(
                f"LR scheduler type changed ({_sched_type_ckpt} → {_sched_type_now}) — "
                "scheduler state not loaded, starting fresh.")
        else:
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
                # Scale inner schedulers (SequentialLR) and eta_min (cosine variants).
                for inner in getattr(sched, "_schedulers", []):
                    inner.base_lrs = [lr * ratio for lr in inner.base_lrs]
                    if hasattr(inner, "eta_min"):
                        inner.eta_min *= ratio
                if hasattr(sched, "eta_min"):
                    sched.eta_min *= ratio
                logger.info(
                    f"LR override: {ckpt_base:.2e} → {new_base:.2e} (ratio {ratio:.4f})"
                )
        # Re-apply selective weight_decay for disc after checkpoint load.
        # With SAN: body gets wd>0 (group 0), conv_post gets wd=0 (group 1).
        # With SN: all groups get wd=0.
        _use_sn = self.config["model"].get("discriminator", {}).get("use_spectral_norm", False)
        _wd = train_cfg.get("weight_decay", 0.01)
        for i, pg in enumerate(self.optim_disc.param_groups):
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
        self._eval_count = ckpt.get("eval_count", 0)
        # If transitioning scheduler types, rebuild schedulers positioned at the
        # current epoch so the LR curve continues smoothly from where training left off.
        if _sched_type_now != _sched_type_ckpt and _sched_type_now in ("cosine", "cosine_warmup_restarts"):
            _tc = self.config["train"]
            _T = _tc["epochs"]
            _cw = _tc.get("cfm_warmup_epochs", 50)
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
                self.sched_flow    = _cosine_at(self.optim_flow,    _T,        self.epoch)
                self.sched_vocoder = _cosine_at(self.optim_vocoder, _T - _cw,  max(0, self.epoch - _cw))
                self.sched_disc    = _cosine_at(self.optim_disc,    _T - _cw,  max(0, self.epoch - _cw))
                logger.info(
                    f"Cosine schedulers rebuilt at epoch {self.epoch} "
                    f"(T_max flow={_T}, voc/disc={_T-_cw})"
                )
            else:  # cosine_warmup_restarts
                _flow_ep = max(0, self.epoch - _wu)
                _disc_ep = max(0, self.epoch - _cw)
                self.sched_flow    = _cosine_at(self.optim_flow,    max(1, _T - _wu),       _flow_ep)
                self.sched_vocoder = _cosine_at(self.optim_vocoder, max(1, _T - _cw - _wu), max(0, self.epoch - _cw - _wu))
                _disc_base = self.optim_disc.param_groups[0]["lr"]
                self.sched_disc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optim_disc, T_0=max(1, _rp), T_mult=_rm,
                    eta_min=_disc_base * _er, last_epoch=_disc_ep)
                logger.info(
                    f"cosine_warmup_restarts schedulers rebuilt at epoch {self.epoch} "
                    f"(T={_T}, cfm_wu={_cw}, lr_wu={_wu}, disc_restart={_rp}x{_rm})"
                )
        logger.info(f"Resumed from {path} (epoch {self.epoch})")


def load_pretrain_for_finetune(
    pretrain_path: str,
    trainer,
    strict: bool = False,
):
    """
    Load a pretrain checkpoint into a fine-tuning trainer.

    Loads flow, vocoder, discriminator, and EMA vocoder weights.
    Skips speaker_embed (different n_speakers) and optimizer states
    (fresh start for fine-tuning).
    """
    ckpt = torch.load(pretrain_path, map_location=trainer.device, weights_only=False)

    trainer.flow.load_state_dict(
        KazeFlowPretrainer._align_state_dict(
            KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["flow"]), trainer.flow), strict=strict)
    trainer.vocoder.load_state_dict(
        KazeFlowPretrainer._align_state_dict(
            KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["vocoder"]), trainer.vocoder), strict=strict)
    trainer.discriminator.load_state_dict(
        KazeFlowPretrainer._align_state_dict(
            KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["discriminator"]), trainer.discriminator), strict=strict)

    # Load EMA weights if available
    if "vocoder_ema" in ckpt and hasattr(trainer, "ema_vocoder"):
        try:
            trainer.ema_vocoder.load_state_dict(
                KazeFlowPretrainer._align_state_dict(
                    KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["vocoder_ema"]),
                    trainer.ema_vocoder))
            logger.info("Loaded EMA vocoder weights from pretrain checkpoint")
        except Exception as e:
            logger.warning(f"Could not load EMA weights: {e} — EMA re-initialized")

    if "flow_ema" in ckpt and hasattr(trainer, "ema_flow"):
        try:
            trainer.ema_flow.load_state_dict(
                KazeFlowPretrainer._align_state_dict(
                    KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["flow_ema"]),
                    trainer.ema_flow))
            logger.info("Loaded EMA flow weights from pretrain checkpoint")
        except Exception as e:
            logger.warning(f"Could not load flow EMA weights: {e} — EMA re-initialized")

    logger.info(
        f"Loaded pretrain weights from {pretrain_path} "
        f"(epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')})"
    )
    logger.info("Speaker embeddings reinitialized for fine-tuning.")
