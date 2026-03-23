"""
KazeFlow v2: Adversarial Flow Matching (AFM) Finetuning.

Same architecture as v1 (FlowEstimator + ChouwaGAN + SAN discriminator).
Only the training procedure changes:

1. CFM receives adversarial gradient: disc → vocoder → mel_hat → flow.
   The flow model learns not just "match the velocity" but also
   "produce mel that sounds real when vocoded."

2. Vocoder trained on curriculum: GT mel → CFM mel_hat transition.
   The vocoder gradually adapts to the CFM's actual outputs,
   eliminating cascaded error by construction.

3. No extra models, no extra parameters. Checkpoints are 100%
   compatible with v1 (same weights, same inference pipeline).

Key design:
    mel_hat = x_t + (1 - t) · v_hat   (single-step prediction of x₁)
    wav_hat = Vocoder(mel_hat)
    loss_afm = GenLoss(Discriminator(wav_hat))
    loss_cfm = velocity_loss + t² · c_afm · loss_afm

    The t² weighting ensures adversarial signal only fires when
    mel_hat is good enough (high t ≈ close to target).
"""

import logging
import math
import random

import numpy as np
import torch

from tqdm import tqdm

from kazeflow.train.trainer import KazeFlowTrainer, _log_section, plot_spectrogram_to_numpy
from kazeflow.train.dataset import create_dataloader
from kazeflow.train.losses import (
    feature_loss,
    mel_spectrogram_loss,
    multi_resolution_stft_loss,
    r1_gradient_penalty,
)


logger = logging.getLogger("kazeflow.trainer_v2")


class KazeFlowV2Trainer(KazeFlowTrainer):
    """
    Adversarial Flow Matching finetuner.

    Inherits all model creation, checkpointing, eval inference from v1.
    Overrides only the training loop to add the adversarial CFM path
    and vocoder curriculum.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        afm = self.config["train"].get("afm", {})
        self.c_afm = afm.get("c_afm", 0.1)
        self.afm_every = max(1, afm.get("adv_every", 2))
        self.afm_ramp_epochs = afm.get("ramp_epochs", 50)
        self.curriculum_epochs = afm.get("curriculum_epochs", 100)

        logger.info(
            f"AFM v2: c_afm={self.c_afm}, adv_every={self.afm_every}, "
            f"ramp_epochs={self.afm_ramp_epochs}, "
            f"curriculum_epochs={self.curriculum_epochs}"
        )

    def _freeze_params(self, *modules):
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(False)

    def _unfreeze_params(self, *modules):
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(True)

    # ── Overridden training loop ─────────────────────────────────────────

    def train(
        self,
        filelist_path: str,
        dataset_root: str,
        resume_path: str = None,
        stop_event=None,
    ):
        """
        AFM finetuning loop — v1 structure with adversarial CFM signal
        and vocoder curriculum.

        Phase 1 (vocoder_warmup_epochs): Mel-only vocoder training (no GAN).
        Phase 2 (remaining epochs): CFM + adversarial path + vocoder curriculum + GAN.
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
            content_embedder=self.config["preprocess"].get("content_embedder", "spin_v2"),
        )

        epochs = train_cfg["epochs"]
        save_every = train_cfg["save_every"]
        log_every = train_cfg["log_every"]

        _log_section("Start (AFM v2 Finetune)")
        logger.info(f"AFM finetuning for {epochs} epochs")
        logger.info(f"Dataset: {len(dataloader.dataset)} samples")
        logger.info(f"Batch size: {train_cfg['batch_size']}")

        for epoch in range(self.epoch, epochs):
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop signal received — exiting training loop.")
                break

            self.epoch = epoch
            _voc_warmup_epochs = train_cfg.get("vocoder_warmup_epochs", 0)
            _disc_active = epoch >= _voc_warmup_epochs

            if _disc_active:
                _phase = "AFM"
                _joint_epoch = epoch - _voc_warmup_epochs

                # GAN ramp
                _gan_ramp_epochs = train_cfg.get("gan_ramp_epochs", 0)
                if _gan_ramp_epochs > 0:
                    _gan_ramp_factor = min(1.0, (_joint_epoch + 1) / _gan_ramp_epochs)
                else:
                    _gan_ramp_factor = 1.0

                # AFM ramp
                _afm_progress = min(1.0, max(0.0, _joint_epoch / max(1, self.afm_ramp_epochs)))
                _effective_c_afm = self.c_afm * _afm_progress

                # Curriculum: GT → CFM mel_hat blend
                _curriculum_progress = min(1.0, max(0.0, _joint_epoch / max(1, self.curriculum_epochs)))
            else:
                _phase = "VocWarmup"
                _gan_ramp_factor = 0.0
                _effective_c_afm = 0.0
                _curriculum_progress = 0.0

            self.flow.train()
            self.vocoder.train()
            self.discriminator.train()

            pbar = tqdm(
                dataloader,
                desc=f"[{_phase}] Epoch {epoch+1}/{epochs}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )
            _accum_cfm = 0
            _gn_flow = torch.tensor(0.0)
            _afm_loss_accum = 0.0
            _afm_loss_count = 0

            # Progressive ODE: precompute effective max for this epoch
            _ode_min_cfg = train_cfg.get("ode_steps_train_min", train_cfg.get("ode_steps_train", 1))
            _ode_max_cfg = train_cfg.get("ode_steps_train_max", train_cfg.get("ode_steps_train", 1))
            if train_cfg.get("progressive_ode", False) and _ode_min_cfg < _ode_max_cfg:
                _ramp = train_cfg.get("ode_ramp_epochs", epochs)
                _progress = min(1.0, max(0.0, epoch / max(1, _ramp)))
                _eff_ode_max = max(_ode_min_cfg, int(round(
                    _ode_min_cfg + (_ode_max_cfg - _ode_min_cfg) * _progress)))
            else:
                _eff_ode_max = _ode_max_cfg

            for batch_idx, batch in enumerate(pbar):
                mel, spin, f0, spk_ids, wav_gt = [x.to(self.device) for x in batch]

                # Speaker embedding: (B, gin_ch, 1)
                g = self.speaker_embed(spk_ids).unsqueeze(-1)

                # Mask: all ones for fixed-length segments
                B, _, T = mel.shape
                x_mask = torch.ones(B, 1, T, device=self.device)
                f0_expanded = f0.unsqueeze(1)  # (B, 1, T)

                # ── Step 1: CFM Loss (with optional AFM path) ────────
                if _accum_cfm == 0:
                    self.optim_flow.zero_grad()

                loss_afm = torch.tensor(0.0, device=self.device)

                if _disc_active:
                    # V2: use forward_afm to get single-step mel prediction
                    with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                        loss_vel, mel_hat_afm, t_afm = self.flow.forward_afm(
                            x_1=mel, x_mask=x_mask,
                            content=spin, f0=f0_expanded, g=g,
                        )
                    loss_cfm = loss_vel

                    # ── V2: Adversarial path ─────────────────────────
                    # Freeze vocoder+disc so only flow gets gradient.
                    # Gradient chain: loss_afm → disc → vocoder → mel_hat → v_t → flow
                    _do_afm = (
                        _effective_c_afm > 0
                        and batch_idx % self.afm_every == 0
                    )
                    if _do_afm:
                        self._freeze_params(self.vocoder, self.discriminator)
                        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                            wav_afm = self.vocoder(mel_hat_afm, f0, g=g)
                            _afm_len = min(wav_afm.shape[-1], wav_gt.shape[-1])
                            wav_afm = wav_afm[..., :_afm_len]
                            _, y_d_gs_afm, _, _ = self.discriminator(
                                wav_gt[..., :_afm_len].detach(), wav_afm,
                                compute_fmaps=False)
                            loss_afm = self._gen_loss_fn(y_d_gs_afm)

                        # t² weighting: adversarial signal meaningful only at high t
                        t_weight = (t_afm ** 2).mean()
                        loss_cfm = loss_cfm + _effective_c_afm * t_weight * loss_afm

                        # Log AFM loss immediately
                        _afm_val = loss_afm.item()
                        if math.isfinite(_afm_val):
                            self.writer.add_scalar("loss/afm", _afm_val, self.global_step)
                        _afm_loss_accum += _afm_val if math.isfinite(_afm_val) else 0.0
                        _afm_loss_count += 1

                        self._unfreeze_params(self.vocoder, self.discriminator)
                else:
                    # Vocoder warmup: regular v1 CFM forward
                    with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                        loss_cfm = self.flow(
                            x_1=mel, x_mask=x_mask,
                            content=spin, f0=f0_expanded, g=g,
                        )

                # NaN guard
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

                self.scaler_flow.scale(loss_cfm / self.cfm_grad_accum).backward()
                _accum_cfm += 1
                _do_flow_step = (_accum_cfm == self.cfm_grad_accum)

                if _do_flow_step:
                    self.scaler_flow.unscale_(self.optim_flow)
                    _gn_flow = torch.nn.utils.clip_grad_norm_(
                        self.flow.parameters(), self.grad_clip_flow)
                    self.scaler_flow.step(self.optim_flow)
                    self.ema_flow.update()
                    _accum_cfm = 0

                # ── Step 2: Generate mel from flow (for vocoder) ─────
                if _ode_min_cfg >= _eff_ode_max:
                    _ode_n = _ode_min_cfg
                else:
                    _ode_n = int(round(math.exp(
                        random.uniform(math.log(_ode_min_cfg), math.log(_eff_ode_max)))))

                self.flow.eval()
                with torch.no_grad():
                    mel_hat = self.flow.sample(
                        content=spin, f0=f0_expanded,
                        x_mask=x_mask, g=g,
                        n_steps=_ode_n,
                        method=train_cfg.get("ode_method_train", "euler"),
                    )
                self.flow.train()

                # ── Step 3: Vocoder (mel → waveform) ─────────────────
                # V2 CHANGE: Curriculum vocoder input — smooth blend
                # instead of binary gt_mel_ratio.
                # At start: 100% GT. At end: 100% mel_hat from CFM.
                _voc_input_mel = (
                    (1.0 - _curriculum_progress) * mel.detach()
                    + _curriculum_progress * mel_hat.detach()
                )

                # Vocoder forward
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    wav_hat = self.vocoder(_voc_input_mel, f0, g=g.detach())

                    # Align lengths
                    target_len = min(wav_hat.shape[-1], wav_gt.shape[-1])
                    wav_hat = wav_hat[..., :target_len]
                    wav_real_detached = wav_gt[..., :target_len].detach()

                # ── Step 4: Discriminator ────────────────────────────
                gn_disc = None
                d_real_mean = d_fake_mean = 0.0
                loss_disc = torch.tensor(0.0, device=self.device)
                loss_lecam = torch.tensor(0.0, device=self.device)
                _apply_r1 = False
                for _disc_i in range(self.n_disc_steps if _disc_active else 0):
                    self.optim_disc.zero_grad()
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

                    # LeCam regularization
                    loss_lecam = torch.tensor(0.0, device=self.device)
                    if self.lecam is not None:
                        loss_lecam = self.lecam.penalty(y_d_rs, y_d_gs)
                        self.lecam.update(y_d_rs, y_d_gs)
                        loss_disc = loss_disc + self.c_lecam * loss_lecam

                    # R1 penalty
                    if _apply_r1:
                        r1_pen = r1_gradient_penalty(y_d_rs, wav_real_detached)
                        loss_disc = loss_disc + (self.c_r1 / 2) * self.r1_interval * r1_pen
                        wav_real_detached.requires_grad_(False)

                    # NaN guard
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
                    d_real_mean, d_fake_mean = _d_real, _d_fake

                # ── Step 5: Generator (vocoder) losses ───────────────
                self.optim_vocoder.zero_grad()

                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    if _disc_active:
                        y_d_rs, y_d_gs, fmap_rs, fmap_gs = self.discriminator(
                            wav_real_detached, wav_hat, compute_fmaps=True,
                        )
                        loss_gen = self._gen_loss_fn(y_d_gs)
                        loss_fm = feature_loss(fmap_rs, fmap_gs)
                    else:
                        loss_gen = torch.tensor(0.0, device=self.device)
                        loss_fm = torch.tensor(0.0, device=self.device)

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
                        _gan_ramp_factor * self.c_gen * loss_gen
                        + _gan_ramp_factor * self.c_fm * loss_fm
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
                    afm=f"{loss_afm.item():.3f}" if _effective_c_afm > 0 else "off",
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
                    # AFM-specific metrics
                    _afm_avg = _afm_loss_accum / max(1, _afm_loss_count)
                    self.writer.add_scalar("loss/afm_avg", _afm_avg, step)
                    self.writer.add_scalar("afm/c_effective", _effective_c_afm, step)
                    self.writer.add_scalar("afm/curriculum", _curriculum_progress, step)
                    if gn_voc is not None:
                        self.writer.add_scalar("loss/voc_total", loss_voc_total.item(), step)
                    # Discriminator health
                    if gn_disc is not None:
                        self.writer.add_scalar("d_score/real", d_real_mean, step)
                        self.writer.add_scalar("d_score/fake", d_fake_mean, step)
                    # Gradient norms
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
            if stop_event is not None and stop_event.is_set():
                break

            self.sched_flow.step()
            self.sched_vocoder.step()
            if _disc_active:
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
                    self.writer.add_image(
                        "eval/mel_diff",
                        plot_spectrogram_to_numpy(
                            np.abs(mel_orig_np - mel_gen_np)
                        ),
                        self.global_step,
                        dataformats="HWC",
                    )

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

        stopped_early = stop_event is not None and stop_event.is_set()
        if stopped_early:
            logger.info(
                "AFM finetuning stopped by user at epoch %d. No checkpoint saved.",
                self.epoch,
            )
        else:
            self._save_checkpoint(epochs)
            logger.info("AFM finetuning complete.")
        self._cleanup()
