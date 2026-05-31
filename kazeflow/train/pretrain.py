"""
KazeFlow Pretraining Pipeline — Rectified Flow Only.

Trains ONLY the RFM (Rectified Flow Matching) model on a multi-speaker dataset.
The vocoder is NOT trained here — use the Vocoder Training tab for that.

Workflow:
  1. Pretrain: train RFM on many speakers → learns content/F0/speaker → mel mapping
  2. Vocoder Training: train ChouwaGAN using the frozen RFM for mel generation
  3. Fine-tune (trainer.py): joint RFM + vocoder fine-tune on a single speaker

Saves a pretrain checkpoint that can be loaded by the Vocoder trainer and
the fine-tuning trainer.
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
from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from kazeflow.models import build_mel_model
from kazeflow.models.vocoder import EMAGenerator
from kazeflow.train.dataset import create_dataloader

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
    Multi-speaker pretraining for KazeFlow — trains ONLY the RFM model.

    The vocoder and discriminator are NOT involved here.
    This keeps pretraining fast and memory-efficient: only one model needs VRAM.

    Key differences from KazeFlowTrainer (fine-tune):
    - Supports multi-GPU via DDP
    - Trains only RFM — no vocoder, no discriminator, no GAN losses
    - Saves pretrain checkpoint compatible with VocoderTrainer and KazeFlowTrainer
    - No warmup phase: RFM learns directly without staged training
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
                logger.info("TF32+BF16 combined precision enabled (recommended)")
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
        elif precision in ("fp32_fp16", "tf32_fp16"):
            _enable_tf32()
            self.use_amp = True
            self.amp_dtype = torch.float16
            logger.info("FP16 AMP enabled")
        elif precision == "fp32":
            logger.info("Using FP32 precision")
        else:
            logger.warning(f"Unknown precision '{precision}', falling back to FP32")

        self._needs_scaler = (self.use_amp and self.amp_dtype == torch.float16)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")
        if _has_tf32():
            torch.set_float32_matmul_precision("high")
            logger.info("float32 matmul precision set to 'high' (TF32, Ampere+)")

        # ── RFM Model (only model we train here) ─────────────────────────
        _log_section("Model")
        self.flow = build_mel_model(
            model_cfg.get("architecture", "rfm"),
            **model_cfg["flow_matching"],
        ).to(self.device)

        n_speakers = model_cfg.get("n_speakers", 1)
        spk_dim = model_cfg.get("speaker_embed_dim", 256)
        self.speaker_embed = nn.Embedding(n_speakers, spk_dim).to(self.device)

        n_params = sum(p.numel() for p in self.flow.parameters() if p.requires_grad)
        logger.info(f"RFM parameters: {n_params / 1e6:.1f}M")

        # ── torch.compile ────────────────────────────────────────────────
        _log_section("Compilation")
        if train_cfg.get("torch_compile", False):
            compile_mode = train_cfg.get("compile_mode", "default")
            try:
                self.flow.estimator = torch.compile(
                    self.flow.estimator, mode=compile_mode)
                logger.info(f"torch.compile (mode='{compile_mode}') applied to RFM estimator")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # ── DDP wrapping ─────────────────────────────────────────────────
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.flow = DDP(self.flow, device_ids=[rank])
            self.speaker_embed = DDP(self.speaker_embed, device_ids=[rank])

        # ── Optimizer ────────────────────────────────────────────────────
        lr_flow = train_cfg["learning_rate_flow"]
        betas_flow = tuple(train_cfg.get("betas_flow", [0.95, 0.999]))
        wd = train_cfg.get("weight_decay", 0.01)
        fused = torch.cuda.is_available()

        # Split: no weight decay on biases, norms, and zero-init out_proj
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

        # ── Scheduler ────────────────────────────────────────────────────
        _total_epochs = train_cfg["epochs"]
        _lr_sched_type = train_cfg.get("lr_scheduler", "cosine")
        _eta_min_ratio = train_cfg.get("lr_eta_min_ratio", 0.01)
        _lr_warmup_epochs = train_cfg.get("lr_warmup_epochs", 0)

        if _lr_sched_type == "cosine":
            self.sched_flow = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim_flow, T_max=max(1, _total_epochs),
                eta_min=lr_flow * _eta_min_ratio)
        elif _lr_sched_type == "cosine_warmup_restarts":
            if _lr_warmup_epochs > 0 and _total_epochs > _lr_warmup_epochs:
                _wu = torch.optim.lr_scheduler.LinearLR(
                    self.optim_flow, start_factor=0.01, end_factor=1.0,
                    total_iters=_lr_warmup_epochs)
                _cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optim_flow, T_max=max(1, _total_epochs - _lr_warmup_epochs),
                    eta_min=lr_flow * _eta_min_ratio)
                self.sched_flow = torch.optim.lr_scheduler.SequentialLR(
                    self.optim_flow, schedulers=[_wu, _cos],
                    milestones=[_lr_warmup_epochs])
            else:
                self.sched_flow = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optim_flow, T_max=max(1, _total_epochs),
                    eta_min=lr_flow * _eta_min_ratio)
        else:
            lr_decay = train_cfg.get("lr_decay", 0.9999)
            self.sched_flow = torch.optim.lr_scheduler.ExponentialLR(
                self.optim_flow, gamma=lr_decay)
        logger.info(f"LR scheduler: {_lr_sched_type}")

        # ── AMP GradScaler ───────────────────────────────────────────────
        self.scaler_flow = torch.amp.GradScaler("cuda", enabled=self._needs_scaler)

        # ── Logging / state ──────────────────────────────────────────────
        if self.is_main:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"), max_queue=1)
        self.global_step = 0
        self.epoch = 0
        self._eval_count = 0

        self.sample_rate = model_cfg["sample_rate"]
        self.hop_length = model_cfg["hop_length"]
        self.n_mels = model_cfg["n_mels"]
        self.segment_frames = model_cfg["segment_frames"]
        self.grad_clip_flow = train_cfg.get("grad_clip_flow", 3.0)
        self.cfm_grad_accum = max(1, int(train_cfg.get("cfm_grad_accum", 1)))

        # ── EMA ──────────────────────────────────────────────────────────
        ema_decay = train_cfg.get("ema_decay", 0.999)
        self.ema_flow = EMAGenerator(
            self._unwrap_model(self.flow), decay=ema_decay)
        self.ema_flow.to(self.device)
        logger.info(f"EMA flow initialized (decay={ema_decay})")

        # Reference sample (lazy)
        self._reference = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _unwrap_model(self, model):
        """Unwrap DDP and torch.compile wrappers."""
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _get_reference_sample(self, dataloader):
        """Get a deterministic full-length reference sample for eval."""
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

        if spin.shape[1] < mel.shape[1]:
            spin = torch.nn.functional.interpolate(
                spin.unsqueeze(0), size=mel.shape[1], mode="linear",
                align_corners=False,
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
        logger.info(
            f"Eval reference: {stem} ({min_len} frames, "
            f"{min_len * self.hop_length / self.sample_rate:.1f}s)"
        )
        return self._reference

    def _eval_infer(self, mel_ref, spin_ref, f0_ref, spk_id_ref):
        """
        Run inference with the EMA flow model for evaluation.
        Returns (mel_hat, ode_steps_used).
        """
        ema_flow = self.ema_flow.get_model()
        ema_flow.eval()
        speaker_embed = self._unwrap_model(self.speaker_embed)

        with torch.no_grad():
            g = speaker_embed(spk_id_ref).unsqueeze(-1)
            B, _, T = mel_ref.shape
            x_mask = torch.ones(B, 1, T, device=self.device)
            f0_expanded = f0_ref.unsqueeze(1)

            infer_cfg = self.config.get("inference", {})
            n_steps = infer_cfg.get("ode_steps", 4)
            method = infer_cfg.get("ode_method", "midpoint")
            mel_hat = ema_flow.sample(
                content=spin_ref,
                f0=f0_expanded,
                x_mask=x_mask,
                g=g,
                n_steps=n_steps,
                method=method,
            )
        return mel_hat, n_steps

    # ── Main Training Loop ────────────────────────────────────────────────

    def train(
        self,
        filelist_path: str,
        dataset_root: str,
        resume_path: str = None,
        stop_event=None,
    ):
        """
        Main pretraining loop — trains only the RFM model.

        No vocoder, no GAN losses, no warmup phases.
        Just clean RFM loss → optim_flow → ema_flow.
        """
        train_cfg = self.config["train"]
        epochs = train_cfg["epochs"]
        save_every = train_cfg["save_every"]
        log_every = train_cfg["log_every"]

        if resume_path:
            self._load_checkpoint(resume_path)

        # Dataset — load without GT waveform (not needed for RFM-only training)
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
            skip_wav=True,  # No waveform needed — we only train RFM
            content_embedder=self.config["preprocess"].get("content_embedder", "rspin"),
        )

        if self.is_main:
            _log_section("Start")
            logger.info(f"Architecture: {self.config['model'].get('architecture', 'rfm').upper()}")
            logger.info(f"Pretraining RFM-only for {epochs} epochs")
            logger.info(f"Dataset: {len(dataloader.dataset)} samples")
            logger.info(f"Batch size: {train_cfg['batch_size']}")
            logger.info(f"Grad accumulation: {self.cfm_grad_accum}")

        for epoch in range(self.epoch, epochs):
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop signal received — exiting training loop.")
                break

            self.epoch = epoch
            self.flow.train()

            pbar = tqdm(
                dataloader,
                desc=f"[Pretrain RFM] Epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
                disable=not self.is_main,
            )
            _accum_cfm = 0
            _gn_flow = torch.tensor(0.0)

            for batch_idx, batch in enumerate(pbar):
                # batch = (mel, spin, f0, spk_ids) — no wav_gt (skip_wav=True)
                mel, spin, f0, spk_ids = batch[0], batch[1], batch[2], batch[3]
                mel = mel.to(self.device)
                spin = spin.to(self.device)
                f0 = f0.to(self.device)
                spk_ids = spk_ids.to(self.device)

                g = self._unwrap_model(self.speaker_embed)(spk_ids).unsqueeze(-1)
                B, _, T = mel.shape
                x_mask = torch.ones(B, 1, T, device=self.device)
                f0_expanded = f0.unsqueeze(1)   # (B, 1, T)

                # Zero grad at start of accumulation window
                if _accum_cfm == 0:
                    self.optim_flow.zero_grad()

                # ── RFM forward pass ──────────────────────────────────────
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    loss_rfm = self.flow(
                        x_1=mel, x_mask=x_mask,
                        content=spin, f0=f0_expanded, g=g,
                    )

                # NaN guard
                if not torch.isfinite(loss_rfm):
                    logger.warning(
                        f"Step {self.global_step}: loss_rfm={loss_rfm.item():.4f}, skipping"
                    )
                    self.optim_flow.zero_grad(set_to_none=True)
                    _accum_cfm = 0
                    self.global_step += 1
                    if stop_event is not None and stop_event.is_set():
                        break
                    continue

                # Backward + accumulation
                self.scaler_flow.scale(loss_rfm / self.cfm_grad_accum).backward()
                _accum_cfm += 1
                _do_step = _accum_cfm == self.cfm_grad_accum

                if _do_step:
                    self.scaler_flow.unscale_(self.optim_flow)
                    _gn_flow = torch.nn.utils.clip_grad_norm_(
                        list(self.flow.parameters()) +
                        list(self._unwrap_model(self.speaker_embed).parameters()),
                        self.grad_clip_flow,
                    )
                    self.scaler_flow.step(self.optim_flow)
                    self.scaler_flow.update()
                    self.ema_flow.update()
                    _accum_cfm = 0

                self.global_step += 1
                pbar.set_postfix(rfm=f"{loss_rfm.item():.4f}",
                                 lr=f"{self.optim_flow.param_groups[0]['lr']:.2e}")

                if self.is_main and self.global_step % log_every == 0:
                    self.writer.add_scalar("loss/rfm", loss_rfm.item(), self.global_step)
                    self.writer.add_scalar("grad_norm/flow", _gn_flow.item(), self.global_step)
                    self.writer.add_scalar("lr/flow", self.optim_flow.param_groups[0]["lr"], self.global_step)

                if stop_event is not None and stop_event.is_set():
                    break

            # ── End of epoch ─────────────────────────────────────────────
            if stop_event is not None and stop_event.is_set():
                break

            self.sched_flow.step()

            if self.is_main and (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch + 1)

                # ── Eval inference → TensorBoard ──────────────────────
                try:
                    ref_mel, ref_spin, ref_f0, ref_spk = \
                        self._get_reference_sample(dataloader)
                    mel_hat_eval, _n_steps = self._eval_infer(
                        ref_mel, ref_spin, ref_f0, ref_spk)

                    mel_orig_np = ref_mel[0].detach().cpu().float().numpy()
                    mel_gen_np = mel_hat_eval[0].detach().cpu().float().numpy()

                    self.writer.add_image(
                        "eval/mel_original",
                        plot_spectrogram_to_numpy(mel_orig_np),
                        self.global_step, dataformats="HWC",
                    )
                    self.writer.add_image(
                        "eval/mel_generated",
                        plot_spectrogram_to_numpy(mel_gen_np),
                        self.global_step, dataformats="HWC",
                    )
                    self.writer.add_image(
                        "eval/mel_diff",
                        plot_spectrogram_to_numpy(np.abs(mel_orig_np - mel_gen_np)),
                        self.global_step, dataformats="HWC",
                    )
                    self.writer.add_scalar("eval/ode_steps", _n_steps, self.global_step)
                    self.writer.flush()
                    logger.info(
                        f"Eval mel logged at epoch {epoch+1}, step {self.global_step}"
                    )
                except Exception as e:
                    logger.warning(f"Eval inference failed: {e}")

        if self.is_main:
            if stop_event is not None and stop_event.is_set():
                logger.info(f"Pretraining stopped at epoch {self.epoch}.")
            else:
                self._save_checkpoint(epochs)
                logger.info("RFM Pretraining complete.")
        self._cleanup()

    # ── Checkpoint ───────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int):
        """Save RFM-only pretrain checkpoint."""
        path = self.output_dir / f"pretrain_{epoch}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        ckpt_data = {
            "epoch": epoch,
            "global_step": self.global_step,
            "flow": self._normalize_checkpoint_sd(
                self._unwrap_model(self.flow).state_dict()),
            "flow_ema": self._normalize_checkpoint_sd(
                self.ema_flow.state_dict()),
            "speaker_embed": self._unwrap_model(self.speaker_embed).state_dict(),
            "optim_flow": self.optim_flow.state_dict(),
            "sched_flow": self.sched_flow.state_dict(),
            "scaler_flow": self.scaler_flow.state_dict(),
            "lr_scheduler": self.config["train"].get("lr_scheduler", "cosine"),
            "eval_count": self._eval_count,
            "is_pretrain": True,
            "pretrain_type": "rfm_only",
        }
        torch.save(ckpt_data, tmp_path)
        tmp_path.replace(path)
        logger.info(f"Saved RFM pretrain checkpoint: {path}")

    @staticmethod
    def _normalize_checkpoint_sd(state_dict: dict) -> dict:
        """Strip _orig_mod. prefix from torch.compile'd checkpoints."""
        if state_dict and all(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
        state_dict = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
        _DERIVED_SUFFIXES = ("_shared_window",)
        return {
            k: v for k, v in state_dict.items()
            if not any(k.endswith(s) for s in _DERIVED_SUFFIXES)
        }

    @staticmethod
    def _align_state_dict(clean_sd: dict, model) -> dict:
        """Re-introduce ._orig_mod. infixes for compiled model loading."""
        def _clean(k): return k.replace("._orig_mod.", ".")
        clean_to_model = {_clean(k): k for k in model.state_dict()}
        return {clean_to_model.get(_clean(k), k): v for k, v in clean_sd.items()}

    def _load_checkpoint(self, path: str):
        """Resume RFM pretraining from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        flow = self._unwrap_model(self.flow)
        _flow_sd = self._align_state_dict(
            self._normalize_checkpoint_sd(ckpt["flow"]), flow)
        _result = flow.load_state_dict(_flow_sd, strict=False)
        if _result.missing_keys:
            logger.info(f"Flow load — new params init: {_result.missing_keys}")

        if "flow_ema" in ckpt:
            self.ema_flow.load_state_dict(
                self._align_state_dict(
                    self._normalize_checkpoint_sd(ckpt["flow_ema"]),
                    self.ema_flow))
        else:
            logger.warning("No flow EMA in checkpoint — EMA re-initialized from flow weights")

        spk = self._unwrap_model(self.speaker_embed)
        try:
            spk.load_state_dict(self._normalize_checkpoint_sd(ckpt["speaker_embed"]))
        except Exception as e:
            logger.warning(f"Speaker embed load failed ({e}) — re-initialized")

        self.optim_flow.load_state_dict(ckpt["optim_flow"])

        _sched_type_now = self.config["train"].get("lr_scheduler", "cosine")
        _sched_type_ckpt = ckpt.get("lr_scheduler", "cosine")
        if _sched_type_now == _sched_type_ckpt and "sched_flow" in ckpt:
            self.sched_flow.load_state_dict(ckpt["sched_flow"])
        else:
            logger.warning(
                f"LR scheduler changed ({_sched_type_ckpt} → {_sched_type_now}) "
                "— scheduler state not restored, starting fresh."
            )

        if "scaler_flow" in ckpt:
            self.scaler_flow.load_state_dict(ckpt["scaler_flow"])

        # Apply any config LR override
        train_cfg = self.config["train"]
        new_lr = train_cfg["learning_rate_flow"]
        ckpt_lr = self.sched_flow.base_lrs[0] if hasattr(self.sched_flow, "base_lrs") \
                  else new_lr
        if abs(ckpt_lr - new_lr) > 1e-12:
            ratio = new_lr / ckpt_lr
            if hasattr(self.sched_flow, "base_lrs"):
                self.sched_flow.base_lrs = [lr * ratio for lr in self.sched_flow.base_lrs]
                self.sched_flow._last_lr = [lr * ratio for lr in self.sched_flow._last_lr]
            for pg in self.optim_flow.param_groups:
                pg["lr"] *= ratio
            logger.info(f"LR override: {ckpt_lr:.2e} → {new_lr:.2e}")

        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self._eval_count = ckpt.get("eval_count", 0)
        logger.info(f"Resumed from {path} (epoch {self.epoch}, step {self.global_step})")

    # ── Cleanup ───────────────────────────────────────────────────────────

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
# Loader for fine-tuning (used by VocoderTrainer and KazeFlowTrainer)
# ---------------------------------------------------------------------------

def load_pretrain_for_finetune(
    pretrain_path: str,
    trainer,
    strict: bool = False,
):
    """
    Load an RFM pretrain checkpoint into a fine-tuning or vocoder trainer.

    Loads: flow weights + EMA flow weights.
    Skips: speaker_embed (different n_speakers), vocoder, discriminator,
           optimizer states (fresh start for fine-tuning).
    """
    ckpt = torch.load(pretrain_path, map_location=trainer.device, weights_only=False)

    # Load flow
    if hasattr(trainer, "flow"):
        _flow_sd = KazeFlowPretrainer._align_state_dict(
            KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["flow"]),
            trainer.flow)
        _result = trainer.flow.load_state_dict(_flow_sd, strict=strict)
        if _result.missing_keys:
            logger.info(f"Flow fine-tune load — new params: {_result.missing_keys}")

    # Load EMA flow
    if "flow_ema" in ckpt and hasattr(trainer, "ema_flow"):
        try:
            trainer.ema_flow.load_state_dict(
                KazeFlowPretrainer._align_state_dict(
                    KazeFlowPretrainer._normalize_checkpoint_sd(ckpt["flow_ema"]),
                    trainer.ema_flow))
            logger.info("Loaded EMA flow weights from pretrain checkpoint")
        except Exception as e:
            logger.warning(f"Could not load EMA flow weights: {e} — EMA re-initialized")

    logger.info(
        f"Loaded RFM pretrain weights from {pretrain_path} "
        f"(epoch {ckpt.get('epoch', '?')}, step {ckpt.get('global_step', '?')})"
    )
    logger.info("Speaker embeddings reinitialized for fine-tuning.")
