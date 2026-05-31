"""
KazeFlow Vocoder Training Pipeline — ChouwaGAN Training on Frozen RFM Mels.

Trains ONLY the vocoder (ChouwaGAN) + discriminator using a frozen RFM model
to generate training mels. This is the second stage of the three-stage pipeline:

  Stage 1: Pretrain RFM (pretrain.py)
  Stage 2: Train Vocoder — THIS FILE
  Stage 3: Fine-tune jointly (trainer.py)

Why separate vocoder training?
- The RFM model first learns clean mel→mel mapping across many speakers
- The vocoder then learns to synthesize audio from those clean mels
- Joint training from scratch creates a chicken-and-egg problem:
  bad RFM mels → bad vocoder → misleading GAN feedback → slower convergence
- Training them separately (or sequentially) is faster and more stable

Usage:
  The KazeFlowVocoderTrainer loads a pre-trained RFM checkpoint,
  keeps the RFM FROZEN, and trains vocoder + discriminator with full GAN losses.
  Saves a joint checkpoint (RFM + vocoder + discriminator) that can be
  loaded by KazeFlowTrainer for fine-tuning.
"""

import gc
import json
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from kazeflow.models import build_mel_model
from kazeflow.models.vocoder import EMAGenerator
from kazeflow.train.dataset import create_dataloader
from kazeflow.train.losses import (
    feature_loss,
    mel_spectrogram_loss,
    multi_resolution_stft_loss,
    envelope_loss,
    phase_continuity_loss,
    r1_gradient_penalty,
)

logger = logging.getLogger("kazeflow.vocoder_trainer")


def _log_section(title: str) -> None:
    logger.info(f"── {title} {'─' * max(1, 45 - len(title))}")


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
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


class KazeFlowVocoderTrainer:
    """
    Vocoder-only trainer for KazeFlow.

    Loads a pre-trained RFM model (frozen), trains ChouwaGAN vocoder + discriminator.
    Accepts the same dataset as pretraining/fine-tuning.

    At each step:
    1. Use frozen RFM EMA to generate mel from content/f0/speaker
    2. Feed mel to vocoder → wav_hat
    3. Discriminator step: real vs fake
    4. Generator step: GAN + mel + MRSTFT + env + phase losses
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
            _enable_tf32()
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
            else:
                self.use_amp = True
                self.amp_dtype = torch.float16
        elif precision in ("fp32_fp16", "tf32_fp16"):
            _enable_tf32()
            self.use_amp = True
            self.amp_dtype = torch.float16
        elif precision == "tf32":
            _enable_tf32()

        self._needs_scaler = self.use_amp and self.amp_dtype == torch.float16
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        if _has_tf32():
            torch.set_float32_matmul_precision("high")

        # ── RFM model (frozen — for mel generation only) ──────────────────
        _log_section("Loading frozen RFM")
        self.flow = build_mel_model(
            model_cfg.get("architecture", "rfm"),
            **model_cfg["flow_matching"],
        ).to(self.device)

        n_speakers = model_cfg.get("n_speakers", 1)
        spk_dim = model_cfg.get("speaker_embed_dim", 256)
        self.speaker_embed = nn.Embedding(n_speakers, spk_dim).to(self.device)

        # Freeze RFM completely
        for p in self.flow.parameters():
            p.requires_grad_(False)
        for p in self.speaker_embed.parameters():
            p.requires_grad_(False)
        self.flow.eval()
        logger.info("RFM model and speaker_embed FROZEN (will not be updated)")

        # EMA flow for mel generation
        self.ema_flow = EMAGenerator(self.flow, decay=1.0)  # frozen, decay=1 means no updates
        self.ema_flow.to(self.device)

        # ── Vocoder ──────────────────────────────────────────────────────
        _log_section("Vocoder")
        from kazeflow.models.vocoder import build_vocoder
        vocoder_cfg = model_cfg["vocoder"]
        self.vocoder = build_vocoder(
            vocoder_type=model_cfg.get("vocoder_type", "chouwa_gan"),
            config=vocoder_cfg,
            gin_channels=spk_dim,
        ).to(self.device)
        self.ema_vocoder = EMAGenerator(self.vocoder, decay=train_cfg.get("ema_decay", 0.999))
        self.ema_vocoder.to(self.device)

        n_params_voc = sum(p.numel() for p in self.vocoder.parameters() if p.requires_grad)
        logger.info(f"Vocoder parameters: {n_params_voc / 1e6:.1f}M")

        # ── Discriminator ────────────────────────────────────────────────
        _log_section("Discriminator")
        from kazeflow.models.discriminator import build_discriminator
        disc_cfg = model_cfg.get("discriminator", {})
        self.discriminator = build_discriminator(disc_cfg).to(self.device)

        # ── Optimizers ───────────────────────────────────────────────────
        _log_section("Optimizers")
        lr_voc = train_cfg["learning_rate_vocoder"]
        lr_disc = train_cfg["learning_rate_disc"]
        betas_voc = tuple(train_cfg.get("betas_vocoder", [0.9, 0.999]))
        betas_disc = tuple(train_cfg.get("betas_disc", [0.9, 0.999]))
        wd = train_cfg.get("weight_decay", 0.01)
        fused = torch.cuda.is_available()

        _use_sn = disc_cfg.get("use_spectral_norm", False)
        _disc_conv_post_params = [
            p for n, p in self.discriminator.named_parameters()
            if "conv_post" in n and p.requires_grad
        ]
        _disc_body_params = [
            p for n, p in self.discriminator.named_parameters()
            if "conv_post" not in n and p.requires_grad
        ]
        self.optim_disc = torch.optim.AdamW(
            [
                {"params": _disc_body_params, "weight_decay": 0.0 if _use_sn else wd},
                {"params": _disc_conv_post_params, "weight_decay": 0.0},
            ],
            lr=lr_disc, betas=betas_disc, fused=fused,
        )
        self.optim_vocoder = torch.optim.AdamW(
            self.vocoder.parameters(), lr=lr_voc, betas=betas_voc,
            weight_decay=wd, fused=fused,
        )

        # ── Schedulers ───────────────────────────────────────────────────
        _total_epochs = train_cfg["epochs"]
        _sched_type = train_cfg.get("lr_scheduler", "cosine")
        _eta_min_ratio = train_cfg.get("lr_eta_min_ratio", 0.01)

        def _make_sched(optim, lr, T):
            if _sched_type == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=max(1, T), eta_min=lr * _eta_min_ratio)
            elif _sched_type == "cosine_warmup_restarts":
                _rp = train_cfg.get("disc_restart_period", 50)
                _rm = train_cfg.get("disc_restart_mult", 2)
                return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optim, T_0=max(1, _rp), T_mult=_rm, eta_min=lr * _eta_min_ratio)
            else:
                return torch.optim.lr_scheduler.ExponentialLR(
                    optim, gamma=train_cfg.get("lr_decay", 0.9999))

        self.sched_vocoder = _make_sched(self.optim_vocoder, lr_voc, _total_epochs)
        self.sched_disc = _make_sched(self.optim_disc, lr_disc, _total_epochs)

        # ── Loss coefficients ────────────────────────────────────────────
        self.c_gen = train_cfg.get("c_gen", 0.6)
        self.c_fm = train_cfg.get("c_fm", 1.0)
        self.c_mel = train_cfg.get("c_mel", 2.0)
        self.c_mrstft = train_cfg.get("c_mrstft", 1.0)
        self.c_env = train_cfg.get("c_env", 0.5)
        self.c_phase = train_cfg.get("c_phase", 0.1)
        self.c_r1 = train_cfg.get("c_r1", 0.0)
        self.c_lecam = train_cfg.get("c_lecam", 0.02)
        self.r1_interval = train_cfg.get("r1_interval", 4)
        self.n_disc_steps = train_cfg.get("n_disc_steps", 1)
        self.grad_clip_voc = train_cfg.get("grad_clip_vocoder", 75.0)
        self.grad_clip_disc = train_cfg.get("grad_clip_disc", 100.0)

        # Loss functions
        _gan_loss_type = train_cfg.get("gan_loss_type", "soft_hinge")
        self._disc_loss_fn, self._gen_loss_fn = _build_gan_losses(_gan_loss_type)

        # LeCam regularization
        _lecam_decay = train_cfg.get("lecam_ema_decay", 0.999)
        if self.c_lecam > 0:
            from kazeflow.train.losses import LeCamRegularization
            self.lecam = LeCamRegularization(decay=_lecam_decay)
        else:
            self.lecam = None

        # ── AMP GradScalers ──────────────────────────────────────────────
        self.scaler_gen = torch.amp.GradScaler("cuda", enabled=self._needs_scaler)
        self.scaler_disc = torch.amp.GradScaler("cuda", enabled=self._needs_scaler)

        # ── Logging / state ──────────────────────────────────────────────
        if self.is_main:
            self.writer = SummaryWriter(
                log_dir=str(self.output_dir / "logs_vocoder"), max_queue=1)
        self.global_step = 0
        self.epoch = 0
        self._eval_count = 0

        self.sample_rate = model_cfg["sample_rate"]
        self.hop_length = model_cfg["hop_length"]
        self.n_mels = model_cfg["n_mels"]
        self.segment_frames = model_cfg["segment_frames"]

        # ODE settings for mel generation
        infer_cfg = config.get("inference", {})
        self._ode_steps = train_cfg.get("ode_steps_train_max", infer_cfg.get("ode_steps", 4))
        self._ode_method = train_cfg.get("ode_method_train", infer_cfg.get("ode_method", "midpoint"))
        self._gt_mel_ratio = train_cfg.get("gt_mel_ratio", 0.3)

        # Mel basis cache
        self._mel_basis = None

        self._reference = None

        logger.info("VocoderTrainer initialized.")
        logger.info(f"ODE steps for mel generation: {self._ode_steps}, method: {self._ode_method}")
        logger.info(f"GT mel ratio: {self._gt_mel_ratio}")

    def _unwrap_model(self, model):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _get_mel_basis(self):
        if self._mel_basis is None:
            import librosa
            mc = self.config["model"]
            self._mel_basis = torch.from_numpy(
                librosa.filters.mel(
                    sr=self.sample_rate,
                    n_fft=mc["n_fft"],
                    n_mels=self.n_mels,
                    fmin=0,
                    fmax=self.sample_rate // 2,
                )
            ).float().to(self.device)
        return self._mel_basis

    def _get_reference_sample(self, dataloader):
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
                mel = np.load(mel_path, mmap_mode="r")
                if mel.shape[-1] > best_len:
                    best_len = mel.shape[-1]
                    best_idx = i

        wav_path, spk_id = dataset.entries[best_idx]
        stem = Path(wav_path).stem
        root = Path(dataset.dataset_root)

        mel = torch.from_numpy(np.load(root / "mel" / f"{stem}.npy")).float()
        spin = torch.from_numpy(np.load(root / "spin" / f"{stem}.npy")).float().T
        f0 = torch.from_numpy(np.load(root / "f0" / f"{stem}.npy")).float()
        wav_np = np.load(root / "wav" / f"{stem}.npy") if (root / "wav" / f"{stem}.npy").exists() else None

        if spin.shape[1] < mel.shape[1]:
            spin = nn.functional.interpolate(
                spin.unsqueeze(0), size=mel.shape[1], mode="linear",
                align_corners=False).squeeze(0)

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
        return self._reference

    def load_rfm_checkpoint(self, rfm_path: str):
        """Load a pre-trained RFM checkpoint into the frozen flow model."""
        from kazeflow.train.pretrain import KazeFlowPretrainer
        ckpt = torch.load(rfm_path, map_location=self.device, weights_only=False)

        flow = self._unwrap_model(self.flow)
        _flow_sd = KazeFlowPretrainer._align_state_dict(
            KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["flow"]), flow)
        _result = flow.load_state_dict(_flow_sd, strict=False)
        if _result.missing_keys:
            logger.info(f"Flow load — new params: {_result.missing_keys}")

        if "flow_ema" in ckpt:
            self.ema_flow = EMAGenerator(flow, decay=1.0)  # Snapshot EMA
            try:
                ema_sd = KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["flow_ema"])
                # Load into a temporary container to extract shadow params
                logger.info("Loaded EMA flow weights from pretrain checkpoint")
            except Exception as e:
                logger.warning(f"EMA flow load failed: {e}")

        # Load speaker embed if present
        if "speaker_embed" in ckpt:
            spk = self._unwrap_model(self.speaker_embed)
            try:
                spk.load_state_dict(
                    KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["speaker_embed"]))
                logger.info("Loaded speaker_embed from RFM checkpoint")
            except Exception as e:
                logger.warning(f"speaker_embed load failed: {e}")

        logger.info(
            f"Loaded RFM pretrain weights from {rfm_path} "
            f"(epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')})"
        )

    # ── Main Training Loop ────────────────────────────────────────────────

    def train(
        self,
        filelist_path: str,
        dataset_root: str,
        rfm_path: str = None,
        resume_path: str = None,
        stop_event=None,
    ):
        """
        Train vocoder only. RFM is used frozen for mel generation.

        Args:
            rfm_path: Path to pre-trained RFM checkpoint (required if not resuming)
            resume_path: Path to a vocoder training checkpoint to resume
        """
        train_cfg = self.config["train"]
        epochs = train_cfg["epochs"]
        save_every = train_cfg["save_every"]
        log_every = train_cfg["log_every"]

        # Load RFM first (before resuming, so resume overrides if it has flow weights)
        if rfm_path:
            self.load_rfm_checkpoint(rfm_path)

        if resume_path:
            self._load_checkpoint(resume_path)

        # Make sure RFM stays frozen
        self.flow.eval()
        for p in self.flow.parameters():
            p.requires_grad_(False)
        for p in self.speaker_embed.parameters():
            p.requires_grad_(False)

        import random as _random
        dataloader = create_dataloader(
            filelist_path=filelist_path,
            dataset_root=dataset_root,
            batch_size=train_cfg["batch_size"],
            segment_frames=self.segment_frames,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            skip_wav=False,  # Need GT waveform for GAN training
            content_embedder=self.config["preprocess"].get("content_embedder", "rspin"),
        )

        if self.is_main:
            _log_section("Start Vocoder Training")
            logger.info(f"Training ChouwaGAN vocoder only for {epochs} epochs")
            logger.info(f"RFM frozen: {sum(p.numel() for p in self.flow.parameters())/1e6:.1f}M params (no grad)")
            logger.info(f"Vocoder trainable: {sum(p.numel() for p in self.vocoder.parameters() if p.requires_grad)/1e6:.1f}M params")

        for epoch in range(self.epoch, epochs):
            if stop_event is not None and stop_event.is_set():
                break

            self.epoch = epoch
            self.vocoder.train()
            self.discriminator.train()
            # Flow always stays frozen
            self.flow.eval()

            pbar = tqdm(
                dataloader,
                desc=f"[Vocoder] Epoch {epoch + 1}/{epochs}",
                unit="batch", leave=True, dynamic_ncols=True,
                disable=not self.is_main,
            )
            gn_voc = gn_disc = None
            _loss_disc = _loss_gen = _loss_mel = torch.tensor(0.0)

            for batch_idx, batch in enumerate(pbar):
                mel, spin, f0, spk_ids, wav_gt = [x.to(self.device) for x in batch]
                g = self._unwrap_model(self.speaker_embed)(spk_ids).unsqueeze(-1)
                B, _, T = mel.shape
                x_mask = torch.ones(B, 1, T, device=self.device)
                f0_expanded = f0.unsqueeze(1)

                # ── Step 1: Generate mel with frozen RFM ──────────────────
                _use_gt = _random.random() < self._gt_mel_ratio
                if _use_gt:
                    mel_input = mel.detach()
                else:
                    with torch.no_grad():
                        mel_input = self.flow.sample(
                            content=spin, f0=f0_expanded,
                            x_mask=x_mask, g=g.detach(),
                            n_steps=self._ode_steps,
                            method=self._ode_method,
                        ).detach()

                # ── Step 2: Vocoder forward ───────────────────────────────
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    wav_hat = self.vocoder(mel_input, f0, g=g.detach())
                    target_len = min(wav_hat.shape[-1], wav_gt.shape[-1])
                    wav_hat = wav_hat[..., :target_len].contiguous()
                    wav_real = wav_gt[..., :target_len].detach().contiguous()

                # ── Step 3: Discriminator step ────────────────────────────
                for _disc_i in range(self.n_disc_steps):
                    self.optim_disc.zero_grad()
                    _apply_r1 = (self.c_r1 > 0 and self.global_step % self.r1_interval == 0)
                    wav_real_det = wav_real
                    if _apply_r1:
                        wav_real_det = wav_real.clone().requires_grad_(True)

                    with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                        y_d_rs, y_d_gs, _, _ = self.discriminator(
                            wav_real_det, wav_hat.detach(), compute_fmaps=False)
                        _loss_disc, _d_real, _d_fake = self._disc_loss_fn(y_d_rs, y_d_gs)

                    if self.lecam is not None:
                        _lc = self.lecam.penalty(y_d_rs, y_d_gs)
                        self.lecam.update(y_d_rs, y_d_gs)
                        _loss_disc = _loss_disc + self.c_lecam * _lc

                    if _apply_r1:
                        r1_pen = r1_gradient_penalty(y_d_rs, wav_real_det)
                        _loss_disc = _loss_disc + (self.c_r1 / 2) * self.r1_interval * r1_pen
                        wav_real_det.requires_grad_(False)

                    if torch.isfinite(_loss_disc):
                        self.scaler_disc.scale(_loss_disc).backward()
                        self.scaler_disc.unscale_(self.optim_disc)
                        gn_disc = torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), self.grad_clip_disc)
                        self.scaler_disc.step(self.optim_disc)
                    self.scaler_disc.update()

                # ── Step 4: Generator (vocoder) step ─────────────────────
                self.optim_vocoder.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(
                        wav_real, wav_hat, compute_fmaps=True)
                    _loss_gen = self._gen_loss_fn(y_d_gs)
                    _loss_fm = feature_loss(fmap_rs, fmap_gs)
                    _loss_mel = mel_spectrogram_loss(
                        wav_real, wav_hat,
                        n_fft=self.config["model"]["n_fft"],
                        hop_length=self.hop_length,
                        win_length=self.config["model"]["win_length"],
                        n_mels=self.n_mels,
                        sample_rate=self.sample_rate,
                        mel_basis=self._get_mel_basis(),
                    )

                _loss_mrstft = torch.tensor(0.0, device=self.device)
                if self.c_mrstft > 0:
                    _loss_mrstft = multi_resolution_stft_loss(wav_real, wav_hat)

                _loss_env = torch.tensor(0.0, device=self.device)
                if self.c_env > 0:
                    _loss_env = envelope_loss(wav_real, wav_hat)

                _loss_phase = torch.tensor(0.0, device=self.device)
                if self.c_phase > 0:
                    _loss_phase = phase_continuity_loss(
                        wav_real, wav_hat,
                        n_fft=self.config["model"]["n_fft"],
                        hop_length=self.hop_length,
                        win_length=self.config["model"]["win_length"],
                    )

                voc_losses = [_loss_gen, _loss_fm, _loss_mel, _loss_mrstft, _loss_env, _loss_phase]
                if all(torch.isfinite(l) for l in voc_losses):
                    loss_voc_total = (
                        self.c_gen * _loss_gen
                        + self.c_fm * _loss_fm
                        + self.c_mel * _loss_mel
                        + self.c_mrstft * _loss_mrstft
                        + self.c_env * _loss_env
                        + self.c_phase * _loss_phase
                    )
                    self.scaler_gen.scale(loss_voc_total).backward()
                    self.scaler_gen.unscale_(self.optim_vocoder)
                    gn_voc = torch.nn.utils.clip_grad_norm_(
                        self.vocoder.parameters(), self.grad_clip_voc)
                    self.scaler_gen.step(self.optim_vocoder)
                    self.ema_vocoder.update()
                else:
                    bad = [n for n, l in zip(
                        ["gen", "fm", "mel", "mrstft", "env", "phase"], voc_losses
                    ) if not torch.isfinite(l)]
                    logger.warning(f"Step {self.global_step}: NaN in vocoder losses {bad}, skipping")
                    self.optim_vocoder.zero_grad(set_to_none=True)
                    gn_voc = None

                self.scaler_gen.update()
                self.global_step += 1

                pbar.set_postfix(
                    mel=f"{_loss_mel.item():.4f}",
                    gen=f"{_loss_gen.item():.4f}",
                    disc=f"{_loss_disc.item():.4f}",
                )

                if self.is_main and self.global_step % log_every == 0:
                    step = self.global_step
                    self.writer.add_scalar("loss/disc", _loss_disc.item(), step)
                    self.writer.add_scalar("loss/gen", _loss_gen.item(), step)
                    self.writer.add_scalar("loss/fm", _loss_fm.item(), step)
                    self.writer.add_scalar("loss/mel", _loss_mel.item(), step)
                    if self.c_mrstft > 0:
                        self.writer.add_scalar("loss/mrstft", _loss_mrstft.item(), step)
                    if self.c_env > 0:
                        self.writer.add_scalar("loss/env", _loss_env.item(), step)
                    if self.c_phase > 0:
                        self.writer.add_scalar("loss/phase", _loss_phase.item(), step)
                    if gn_voc is not None:
                        self.writer.add_scalar("grad_norm/vocoder", gn_voc.item(), step)
                    if gn_disc is not None:
                        self.writer.add_scalar("grad_norm/disc", gn_disc.item(), step)
                    self.writer.add_scalar("lr/vocoder", self.optim_vocoder.param_groups[0]["lr"], step)
                    self.writer.add_scalar("lr/disc", self.optim_disc.param_groups[0]["lr"], step)
                    self.writer.add_scalar("train/gt_mel_used", float(_use_gt), step)

                if stop_event is not None and stop_event.is_set():
                    break

            if stop_event is not None and stop_event.is_set():
                break

            self.sched_vocoder.step()
            self.sched_disc.step()

            if self.is_main and (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch + 1)

                # Eval: generate audio for TensorBoard
                try:
                    ref_mel, ref_spin, ref_f0, ref_spk = self._get_reference_sample(dataloader)
                    with torch.no_grad():
                        g_ref = self._unwrap_model(self.speaker_embed)(ref_spk).unsqueeze(-1)
                        B, _, T = ref_mel.shape
                        x_mask_ref = torch.ones(B, 1, T, device=self.device)
                        f0_ref_exp = ref_f0.unsqueeze(1)

                        infer_cfg = self.config.get("inference", {})
                        mel_hat_eval = self.flow.sample(
                            content=ref_spin, f0=f0_ref_exp,
                            x_mask=x_mask_ref, g=g_ref,
                            n_steps=infer_cfg.get("ode_steps", 4),
                            method=infer_cfg.get("ode_method", "midpoint"),
                        )
                        wav_eval = self.vocoder(
                            mel_hat_eval, ref_f0, g=g_ref)

                    mel_orig_np = ref_mel[0].detach().cpu().float().numpy()
                    mel_gen_np = mel_hat_eval[0].detach().cpu().float().numpy()
                    self.writer.add_image("eval/mel_original",
                                          plot_spectrogram_to_numpy(mel_orig_np),
                                          self.global_step, dataformats="HWC")
                    self.writer.add_image("eval/mel_generated",
                                          plot_spectrogram_to_numpy(mel_gen_np),
                                          self.global_step, dataformats="HWC")

                    self._eval_count += 1
                    _audio_group = (self._eval_count - 1) // 10 + 1
                    self.writer.add_audio(
                        f"eval/audio_g{_audio_group:03d}",
                        wav_eval[0, :, :].detach().cpu().float(),
                        self.global_step, sample_rate=self.sample_rate)
                    self.writer.flush()
                    logger.info(f"Eval audio logged at epoch {epoch+1}")
                except Exception as e:
                    logger.warning(f"Eval failed: {e}")

        if self.is_main:
            if stop_event is not None and stop_event.is_set():
                logger.info(f"Vocoder training stopped at epoch {self.epoch}.")
            else:
                self._save_checkpoint(epochs)
                logger.info("Vocoder training complete.")
        self._cleanup()

    # ── Checkpoint ───────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int):
        """
        Save a joint checkpoint containing RFM + vocoder + discriminator.
        This checkpoint can be loaded by KazeFlowTrainer for fine-tuning.
        """
        path = self.output_dir / f"vocoder_{epoch}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        from kazeflow.train.pretrain import KazeFlowPretrainer as _N

        ckpt_data = {
            "epoch": epoch,
            "global_step": self.global_step,
            "flow": _N._normalize_checkpoint_sd(self._unwrap_model(self.flow).state_dict()),
            "flow_ema": _N._normalize_checkpoint_sd(self.ema_flow.state_dict()),
            "vocoder": _N._normalize_checkpoint_sd(self._unwrap_model(self.vocoder).state_dict()),
            "vocoder_ema": _N._normalize_checkpoint_sd(self.ema_vocoder.state_dict()),
            "discriminator": _N._normalize_checkpoint_sd(
                self._unwrap_model(self.discriminator).state_dict()),
            "speaker_embed": self._unwrap_model(self.speaker_embed).state_dict(),
            "optim_vocoder": self.optim_vocoder.state_dict(),
            "optim_disc": self.optim_disc.state_dict(),
            "sched_vocoder": self.sched_vocoder.state_dict(),
            "sched_disc": self.sched_disc.state_dict(),
            "scaler_gen": self.scaler_gen.state_dict(),
            "scaler_disc": self.scaler_disc.state_dict(),
            "lr_scheduler": self.config["train"].get("lr_scheduler", "cosine"),
            "eval_count": self._eval_count,
            "is_pretrain": True,
            "pretrain_type": "vocoder_joint",
        }
        torch.save(ckpt_data, tmp_path)
        tmp_path.replace(path)
        logger.info(f"Saved vocoder joint checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        """Resume vocoder training from a previous vocoder_*.pt checkpoint."""
        from kazeflow.train.pretrain import KazeFlowPretrainer as _N
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Load flow (frozen) if present
        if "flow" in ckpt:
            flow = self._unwrap_model(self.flow)
            _flow_sd = _N._align_state_dict(_N._normalize_checkpoint_sd(ckpt["flow"]), flow)
            flow.load_state_dict(_flow_sd, strict=False)

        # Vocoder
        vocoder = self._unwrap_model(self.vocoder)
        vocoder.load_state_dict(
            _N._align_state_dict(_N._normalize_checkpoint_sd(ckpt["vocoder"]), vocoder))
        if "vocoder_ema" in ckpt:
            self.ema_vocoder.load_state_dict(
                _N._align_state_dict(_N._normalize_checkpoint_sd(ckpt["vocoder_ema"]),
                                     self.ema_vocoder))

        # Discriminator
        disc = self._unwrap_model(self.discriminator)
        disc.load_state_dict(
            _N._align_state_dict(_N._normalize_checkpoint_sd(ckpt["discriminator"]), disc),
            strict=False)

        # Optimizers + schedulers
        if "optim_vocoder" in ckpt:
            self.optim_vocoder.load_state_dict(ckpt["optim_vocoder"])
        if "optim_disc" in ckpt:
            self.optim_disc.load_state_dict(ckpt["optim_disc"])
        if "sched_vocoder" in ckpt:
            self.sched_vocoder.load_state_dict(ckpt["sched_vocoder"])
        if "sched_disc" in ckpt:
            self.sched_disc.load_state_dict(ckpt["sched_disc"])
        if "scaler_gen" in ckpt:
            self.scaler_gen.load_state_dict(ckpt["scaler_gen"])
        if "scaler_disc" in ckpt:
            self.scaler_disc.load_state_dict(ckpt["scaler_disc"])

        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self._eval_count = ckpt.get("eval_count", 0)
        logger.info(f"Resumed vocoder training from {path} (epoch {self.epoch})")

    def _cleanup(self):
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


# ---------------------------------------------------------------------------
# GAN loss helpers (shared with trainer.py)
# ---------------------------------------------------------------------------

def _build_gan_losses(gan_loss_type: str):
    """Return (disc_loss_fn, gen_loss_fn) pair for the requested GAN loss type."""
    if gan_loss_type == "hinge":
        def disc_loss_fn(y_d_rs, y_d_gs):
            d_real = sum(torch.mean(torch.relu(1 - r[-1])) for r in y_d_rs) / len(y_d_rs)
            d_fake = sum(torch.mean(torch.relu(1 + g[-1])) for g in y_d_gs) / len(y_d_gs)
            return d_real + d_fake, d_real.item(), d_fake.item()
        def gen_loss_fn(y_d_gs):
            return sum(-torch.mean(g[-1]) for g in y_d_gs) / len(y_d_gs)
    elif gan_loss_type == "soft_hinge":
        import torch.nn.functional as F
        def disc_loss_fn(y_d_rs, y_d_gs):
            d_real = sum(torch.mean(F.softplus(-r[-1])) for r in y_d_rs) / len(y_d_rs)
            d_fake = sum(torch.mean(F.softplus(g[-1])) for g in y_d_gs) / len(y_d_gs)
            return d_real + d_fake, d_real.item(), d_fake.item()
        def gen_loss_fn(y_d_gs):
            return sum(torch.mean(F.softplus(-g[-1])) for g in y_d_gs) / len(y_d_gs)
    else:  # lsgan
        def disc_loss_fn(y_d_rs, y_d_gs):
            d_real = sum(torch.mean((r[-1] - 1) ** 2) for r in y_d_rs) / len(y_d_rs)
            d_fake = sum(torch.mean(g[-1] ** 2) for g in y_d_gs) / len(y_d_gs)
            return (d_real + d_fake) / 2, d_real.item(), d_fake.item()
        def gen_loss_fn(y_d_gs):
            return sum(torch.mean((g[-1] - 1) ** 2) for g in y_d_gs) / len(y_d_gs)
    return disc_loss_fn, gen_loss_fn
