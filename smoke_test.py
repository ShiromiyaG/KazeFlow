"""
KazeFlow smoke test.

Verifies that every major component (models, dataset, losses, trainer
construction) can be imported, instantiated, and run through a single
forward / loss step without crashing.

Run:
    python smoke_test.py            # auto-selects CPU or CUDA
    python smoke_test.py --device cpu
    python smoke_test.py --device cuda
    python smoke_test.py --full     # also exercises KazeFlowTrainer init
"""

import argparse
import copy
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger("smoke_test")

# ---------------------------------------------------------------------------
# Minimal config (tiny dimensions — keeps the test fast on CPU)
# ---------------------------------------------------------------------------

TINY_CONFIG = {
    "model": {
        "sample_rate": 48000,
        "hop_length": 480,
        "n_fft": 512,
        "win_length": 512,
        "n_mels": 32,
        "segment_size": 480 * 8,
        "segment_frames": 8,
        "speaker_embed_dim": 16,
        "n_speakers": 4,
        "flow_matching": {
            "mel_channels": 32,
            "hidden_channels": 32,
            "cond_channels": 64,
            "gin_channels": 16,
            "n_layers": 2,
            "kernel_size": 3,
            "dilation_rate": 2,
            "dropout": 0.0,
            "sigma_min": 1e-4,
            "t_sampling": "uniform",
            "t_logit_mean": 0.0,
            "t_logit_std": 1.0,
            "cfg_dropout": 0.0,
        },
        "vocoder": {
            "mel_channels": 32,
            "upsample_rates": [8, 5, 6, 2],
            "upsample_initial_channel": 32,
            "upsample_kernel_sizes": [16, 10, 12, 4],
            "resblock_kernel_sizes": [3],
            "resblock_dilation_sizes": [[1, 3]],
            "gin_channels": 16,
            "n_harmonics": 2,
            "use_checkpoint": False,
            "n_fft": 512,
            "win_length": 512,
        },
        "discriminator": {
            "stft_configs": [
                {"n_fft": 512, "hop_length": 128, "win_length": 512},
            ],
            "stft_channels": 8,
            "stft_max_channels": 16,
            "stft_n_layers": 2,
            "mpd_periods": [2],
            "mpd_channels": 8,
            "mpd_max_channels": 16,
            "mpd_n_layers": 2,
            "univhd_n_harmonics": 4,
            "univhd_bins_per_octave": 8,
            "univhd_channels": 8,
            "univhd_n_fft": 512,
            "univhd_hop_length": 128,
            "use_spectral_norm": False,
        },
    },
    "train": {
        "batch_size": 2,
        "learning_rate_flow": 1e-4,
        "learning_rate_vocoder": 2e-4,
        "learning_rate_disc": 2e-4,
        "betas": [0.8, 0.99],
        "betas_flow": [0.95, 0.999],
        "betas_vocoder": [0.9, 0.999],
        "betas_disc": [0.9, 0.999],
        "lr_decay": 0.999,
        "weight_decay": 0.01,
        "epochs": 1,
        "save_every": 1,
        "log_every": 1,
        "grad_clip_flow": 3.0,
        "cfm_grad_accum": 1,
        "grad_clip_vocoder": 50.0,
        "conv_post_grad_clip": 0.0,
        "grad_clip_disc": 10.0,
        "c_gen": 0.1,
        "c_mel": 2.0,
        "c_fm": 2.0,
        "gan_loss_type": "soft_hinge",
        "c_r1": 0.0,
        "r1_interval": 4,
        "c_env": 0.0,
        "c_mrstft": 0.0,
        "c_phase": 0.0,
        "ema_decay": 0.999,
        "use_gradient_balancer": False,
        "balancer_ema_decay": 0.999,
        "ode_steps_train": 1,
        "cfm_warmup_epochs": 0,
        "f0_method": "rmvpe",
        "num_workers": 0,
        "pin_memory": False,
        "seed": 42,
        "precision": "fp32",
        "torch_compile": False,
        "compile_mode": "default",
    },
    "inference": {
        "ode_steps": 4,
        "ode_method": "midpoint",
        "guidance_scale": 1.5,
        "f0_method": "rmvpe",
        "f0_shift": 0,
        "index_rate": 0.0,
    },
    "preprocess": {
        "min_duration": 0.5,
        "max_duration": 15.0,
        "highpass_freq": 48,
        "target_db": -23.0,
        "spin_model": "dr87/spinv2_rvc",
    },
}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    """Run fn(); record pass/fail; return True on success."""
    try:
        fn()
        log.info("%-55s %s", name, PASS)
        _results.append((name, True, ""))
        return True
    except Exception as exc:
        log.error("%-55s %s  (%s: %s)", name, FAIL, type(exc).__name__, exc)
        _results.append((name, False, f"{type(exc).__name__}: {exc}"))
        return False


# ---------------------------------------------------------------------------
# Helpers to build dummy tensors
# ---------------------------------------------------------------------------

def _dummy_batch(cfg: dict, device: torch.device, B: int = 2):
    """Return a batch tuple matching KazeFlowCollator output."""
    m = cfg["model"]
    T = m["segment_frames"]
    T_audio = T * m["hop_length"]
    mel = torch.randn(B, m["n_mels"], T, device=device)
    spin = torch.randn(B, m["flow_matching"]["cond_channels"], T, device=device)
    f0 = torch.rand(B, T, device=device) * 220 + 80   # plausible Hz
    spk_ids = torch.randint(0, m["n_speakers"], (B,), device=device)
    wavs = torch.randn(B, 1, T_audio, device=device)
    return mel, spin, f0, spk_ids, wavs


def _spk_embed(cfg: dict, spk_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build a (B, gin_channels, 1) speaker embedding from raw ids."""
    spk_dim = cfg["model"]["speaker_embed_dim"]
    n_spk = cfg["model"]["n_speakers"]
    emb = nn.Embedding(n_spk, spk_dim).to(device)
    g = emb(spk_ids).unsqueeze(-1)      # (B, spk_dim, 1)
    return g


# ===========================================================================
# Tests
# ===========================================================================

def test_imports():
    """All kazeflow modules can be imported."""
    import kazeflow.models.flow_matching
    import kazeflow.models.vocoder
    import kazeflow.models.discriminator
    import kazeflow.train.dataset
    import kazeflow.train.losses
    import kazeflow.train.trainer
    import kazeflow.train.pretrain


def test_config_load():
    """load_config returns a valid config with required keys."""
    from kazeflow.configs import load_config
    cfg = load_config(48000, preset="pretrain")
    assert "model" in cfg and "train" in cfg and "inference" in cfg
    assert "flow_matching" in cfg["model"]
    assert "vocoder" in cfg["model"]
    assert "discriminator" in cfg["model"]


def test_flow_matching_forward(cfg: dict, device: torch.device):
    """ConditionalFlowMatching forward pass returns a scalar loss."""
    from kazeflow.models.flow_matching import ConditionalFlowMatching
    m = cfg["model"]
    B = 2
    T = m["segment_frames"]
    model = ConditionalFlowMatching(**m["flow_matching"]).to(device).eval()
    mel = torch.randn(B, m["n_mels"], T, device=device)
    spin = torch.randn(B, m["flow_matching"]["cond_channels"], T, device=device)
    f0 = (torch.rand(B, T, device=device) * 220 + 80).unsqueeze(1)  # (B, 1, T)
    x_mask = torch.ones(B, 1, T, device=device)
    g = torch.randn(B, m["flow_matching"]["gin_channels"], 1, device=device)
    with torch.no_grad():
        loss = model(x_1=mel, x_mask=x_mask, content=spin, f0=f0, g=g)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"CFM loss is not finite: {loss.item()}"


def test_flow_matching_sample(cfg: dict, device: torch.device):
    """ConditionalFlowMatching.sample returns correct mel shape."""
    from kazeflow.models.flow_matching import ConditionalFlowMatching
    m = cfg["model"]
    B = 1
    T = m["segment_frames"]
    model = ConditionalFlowMatching(**m["flow_matching"]).to(device).eval()
    spin = torch.randn(B, m["flow_matching"]["cond_channels"], T, device=device)
    f0 = (torch.rand(B, T, device=device) * 220 + 80).unsqueeze(1)   # (B, 1, T)
    x_mask = torch.ones(B, 1, T, device=device)
    g = torch.randn(B, m["flow_matching"]["gin_channels"], 1, device=device)
    with torch.no_grad():
        mel_hat = model.sample(
            content=spin, f0=f0, x_mask=x_mask, g=g,
            n_steps=2, method="euler",
        )
    expected = (B, m["n_mels"], T)
    assert mel_hat.shape == expected, \
        f"Expected mel_hat {expected}, got {tuple(mel_hat.shape)}"


def test_vocoder_forward(cfg: dict, device: torch.device):
    """Vocoder forward returns correct waveform shape."""
    from kazeflow.models.vocoder import build_vocoder
    m = cfg["model"]
    B = 2
    T = m["segment_frames"]
    T_audio = T * m["hop_length"]
    vocoder_type = m.get("vocoder_type", "chouwa_gan")
    model = build_vocoder(
        vocoder_type, sr=m["sample_rate"], **m["vocoder"]
    ).to(device).eval()
    mel = torch.randn(B, m["n_mels"], T, device=device)
    f0 = torch.rand(B, T, device=device) * 220 + 80
    g = torch.randn(B, m["vocoder"]["gin_channels"], 1, device=device)
    with torch.no_grad():
        wav = model(mel, f0, g=g)
    assert wav.shape == (B, 1, T_audio), \
        f"Expected wav shape {(B, 1, T_audio)}, got {tuple(wav.shape)}"
    assert torch.isfinite(wav).all(), "Vocoder output contains NaN/Inf"


def test_ema_generator(cfg: dict, device: torch.device):
    """EMAGenerator wraps a model and .get_model() returns a runnable copy."""
    from kazeflow.models.vocoder import build_vocoder, EMAGenerator
    m = cfg["model"]
    vocoder_type = m.get("vocoder_type", "chouwa_gan")
    vocoder = build_vocoder(vocoder_type, sr=m["sample_rate"], **m["vocoder"]).to(device)
    ema = EMAGenerator(vocoder, decay=0.999)
    ema.to(device)
    # update should run without error
    ema.update()
    ema_model = ema.get_model()
    assert ema_model is not None
    # verify the EMA model runs
    B, T = 1, m["segment_frames"]
    mel = torch.randn(B, m["n_mels"], T, device=device)
    f0 = torch.rand(B, T, device=device) * 220 + 80
    g = torch.randn(B, m["vocoder"]["gin_channels"], 1, device=device)
    ema_model.eval()
    with torch.no_grad():
        wav = ema_model(mel, f0, g=g)
    assert wav.shape[0] == B


def test_discriminator_forward(cfg: dict, device: torch.device):
    """Discriminator forward returns matched lists of scores/fmaps."""
    from kazeflow.models.discriminator import build_discriminator
    m = cfg["model"]
    B = 2
    T_audio = m["segment_frames"] * m["hop_length"]
    disc_type = m.get("discriminator_type", m.get("vocoder_type", "chouwa_gan"))
    model = build_discriminator(
        disc_type,
        sample_rate=m["sample_rate"],
        **m["discriminator"],
    ).to(device).eval()
    y_real = torch.randn(B, 1, T_audio, device=device)
    y_fake = torch.randn(B, 1, T_audio, device=device)
    with torch.no_grad():
        scores_real, scores_fake, fmaps_real, fmaps_fake = model(y_real, y_fake)
    assert len(scores_real) == len(scores_fake), "Mismatched discriminator outputs"
    assert len(fmaps_real) == len(fmaps_fake), "Mismatched feature maps"
    assert len(scores_real) > 0, "No discriminator outputs returned"


def test_losses(cfg: dict, device: torch.device):
    """All imported loss functions execute without error."""
    from kazeflow.train.losses import (
        discriminator_loss_hinge,
        discriminator_loss_lsgan,
        feature_loss,
        generator_loss_hinge,
        generator_loss_lsgan,
        generator_loss_soft_hinge,
        mel_spectrogram_loss,
    )
    from kazeflow.models.discriminator import build_discriminator
    m = cfg["model"]
    B = 2
    T_audio = m["segment_frames"] * m["hop_length"]

    disc_type = m.get("discriminator_type", m.get("vocoder_type", "chouwa_gan"))
    disc = build_discriminator(
        disc_type,
        sample_rate=m["sample_rate"],
        **m["discriminator"],
    ).to(device).eval()
    y_real = torch.randn(B, 1, T_audio, device=device)
    y_fake = torch.randn(B, 1, T_audio, device=device)
    with torch.no_grad():
        scores_real, scores_fake, fmaps_real, fmaps_fake = disc(y_real, y_fake)

    # Discriminator losses
    d_loss, *_ = discriminator_loss_hinge(scores_real, scores_fake)
    assert torch.isfinite(d_loss)
    d_loss2, *_ = discriminator_loss_lsgan(scores_real, scores_fake)
    assert torch.isfinite(d_loss2)

    # Generator losses (return a scalar Tensor, not a tuple)
    g_loss  = generator_loss_hinge(scores_fake)
    assert torch.isfinite(g_loss)
    g_loss2 = generator_loss_lsgan(scores_fake)
    assert torch.isfinite(g_loss2)
    g_loss3 = generator_loss_soft_hinge(scores_fake)
    assert torch.isfinite(g_loss3)

    # Feature matching loss
    fm = feature_loss(fmaps_real, fmaps_fake)
    assert torch.isfinite(fm)

    # Mel loss
    mel_l = mel_spectrogram_loss(
        y_real, y_fake,
        n_fft=m["n_fft"], hop_length=m["hop_length"],
        win_length=m["win_length"], n_mels=m["n_mels"],
        sample_rate=m["sample_rate"],
    )
    assert torch.isfinite(mel_l)


def test_dataset_and_dataloader(cfg: dict):
    """KazeFlowDataset loads from a temp dir and DataLoader yields correct shapes."""
    from kazeflow.train.dataset import KazeFlowDataset, create_dataloader
    m = cfg["model"]
    T = m["segment_frames"] + 4   # slightly longer than segment so crop is exercised
    T_audio = T * m["hop_length"]
    n_spk = m["n_speakers"]

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for d in ("mel", "spin", "f0", "sliced_audios"):
            (root / d).mkdir()

        for i in range(3):
            stem = f"utt{i:03d}"
            # mel: (n_mels, T)
            np.save(root / "mel" / f"{stem}.npy",
                    np.random.randn(m["n_mels"], T).astype(np.float32))
            # spin: (T, cond_channels) — SPIN stores (T, 768) on disk
            cond_ch = m["flow_matching"]["cond_channels"]
            np.save(root / "spin" / f"{stem}.npy",
                    np.random.randn(T, cond_ch).astype(np.float32))
            # f0: (T,)
            np.save(root / "f0" / f"{stem}.npy",
                    (np.random.rand(T) * 220 + 80).astype(np.float32))
            # GT waveform
            import soundfile as sf
            sf.write(
                str(root / "sliced_audios" / f"{stem}.wav"),
                np.random.randn(T_audio).astype(np.float32),
                m["sample_rate"],
            )

        # write filelist
        fl = root / "filelist.txt"
        with open(fl, "w") as f:
            for i in range(3):
                spk = i % n_spk
                f.write(f"utt{i:03d}.wav|{spk}\n")

        dataset = KazeFlowDataset(
            filelist_path=str(fl),
            dataset_root=str(root),
            segment_frames=m["segment_frames"],
            n_mels=m["n_mels"],
            hop_length=m["hop_length"],
            sample_rate=m["sample_rate"],
        )
        assert len(dataset) == 3, f"Expected 3 samples, got {len(dataset)}"

        item = dataset[0]
        assert len(item) == 5, "Expected 5-tuple from __getitem__"
        mel_s, spin_s, f0_s, spk_id_s, wav_s = item
        assert mel_s.shape == (m["n_mels"], m["segment_frames"])
        assert spin_s.shape[0] == cond_ch
        assert f0_s.shape[0] == m["segment_frames"]
        assert wav_s.shape[0] == m["segment_frames"] * m["hop_length"]

        loader = create_dataloader(
            filelist_path=str(fl),
            dataset_root=str(root),
            batch_size=2,
            segment_frames=m["segment_frames"],
            n_mels=m["n_mels"],
            hop_length=m["hop_length"],
            sample_rate=m["sample_rate"],
            num_workers=0,
            pin_memory=False,
        )
        batch = next(iter(loader))
        B_mel, n_mels, T_mel = batch[0].shape
        assert n_mels == m["n_mels"]
        assert T_mel == m["segment_frames"]
        assert batch[1].shape[:2] == (B_mel, cond_ch)
        assert batch[2].shape == (B_mel, m["segment_frames"])
        assert batch[3].dtype == torch.long
        assert batch[4].shape == (B_mel, 1, m["segment_frames"] * m["hop_length"])


def test_trainer_init(cfg: dict, device: str):
    """KazeFlowTrainer can be instantiated without error."""
    from kazeflow.train.trainer import KazeFlowTrainer
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = KazeFlowTrainer(config=cfg, output_dir=tmpdir, device=device)
        assert hasattr(trainer, "flow")
        assert hasattr(trainer, "vocoder")
        assert hasattr(trainer, "discriminator")
        assert hasattr(trainer, "speaker_embed")
        assert hasattr(trainer, "ema_vocoder")
        assert hasattr(trainer, "optim_flow")
        assert hasattr(trainer, "optim_vocoder")
        assert hasattr(trainer, "optim_disc")


def test_pretrainer_init(cfg: dict, device: str):
    """KazeFlowPretrainer can be instantiated without error."""
    from kazeflow.train.pretrain import KazeFlowPretrainer
    with tempfile.TemporaryDirectory() as tmpdir:
        pt = KazeFlowPretrainer(
            config=cfg, output_dir=tmpdir,
            device=device, rank=0, world_size=1,
        )
        assert hasattr(pt, "flow")
        assert hasattr(pt, "vocoder")
        assert hasattr(pt, "discriminator")
        assert hasattr(pt, "speaker_embed")
        assert hasattr(pt, "ema_vocoder")


def test_full_training_step(cfg: dict, device: torch.device):
    """
    One complete forward+backward step through the Trainer's internal logic:
    CFM loss + vocoder GAN losses without touching disk.
    """
    from kazeflow.models.flow_matching import ConditionalFlowMatching
    from kazeflow.models.vocoder import build_vocoder, EMAGenerator
    from kazeflow.models.discriminator import build_discriminator
    from kazeflow.train.losses import (
        generator_loss_soft_hinge, discriminator_loss_hinge,
        feature_loss, mel_spectrogram_loss,
    )

    m = cfg["model"]
    B = 2
    T = m["segment_frames"]
    T_audio = T * m["hop_length"]
    cond_ch = m["flow_matching"]["cond_channels"]
    gin_ch  = m["flow_matching"]["gin_channels"]

    vocoder_type = m.get("vocoder_type", "chouwa_gan")
    disc_type = m.get("discriminator_type", vocoder_type)
    flow       = ConditionalFlowMatching(**m["flow_matching"]).to(device)
    vocoder    = build_vocoder(vocoder_type, sr=m["sample_rate"], **m["vocoder"]).to(device)
    disc       = build_discriminator(disc_type, sample_rate=m["sample_rate"], **m["discriminator"]).to(device)
    spk_emb    = nn.Embedding(m["n_speakers"], m["speaker_embed_dim"]).to(device)
    ema_voc    = EMAGenerator(vocoder, decay=0.999); ema_voc.to(device)

    opt_flow = torch.optim.AdamW(flow.parameters(), lr=1e-4)
    opt_voc  = torch.optim.AdamW(vocoder.parameters(), lr=2e-4)
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=2e-4)

    mel    = torch.randn(B, m["n_mels"], T, device=device)
    spin   = torch.randn(B, cond_ch, T, device=device)
    f0     = torch.rand(B, T, device=device) * 220 + 80
    spk_ids = torch.randint(0, m["n_speakers"], (B,), device=device)
    wav_gt  = torch.randn(B, 1, T_audio, device=device)

    g      = spk_emb(spk_ids).unsqueeze(-1)   # (B, spk_dim, 1)
    x_mask = torch.ones(B, 1, T, device=device)
    f0_exp = f0.unsqueeze(1)                   # (B, 1, T)

    # ── CFM loss ──────────────────────────────────────────────────────────
    flow.train()
    opt_flow.zero_grad()
    cfm_loss = flow(x_1=mel, x_mask=x_mask, content=spin, f0=f0_exp, g=g)
    assert torch.isfinite(cfm_loss), f"CFM loss non-finite: {cfm_loss}"
    cfm_loss.backward()
    opt_flow.step()

    # ── Vocoder (generator) step ──────────────────────────────────────────
    vocoder.train()
    opt_voc.zero_grad()
    wav_fake = vocoder(mel.detach(), f0.detach(), g=g.detach())
    assert wav_fake.shape == (B, 1, T_audio), \
        f"Vocoder output {wav_fake.shape} != {(B, 1, T_audio)}"

    _, scores_fake_g, fmaps_r, fmaps_g = disc(wav_gt, wav_fake)
    g_adv     = generator_loss_soft_hinge(scores_fake_g)
    g_fm       = feature_loss(fmaps_r, fmaps_g)
    g_mel      = mel_spectrogram_loss(
        wav_gt, wav_fake,
        n_fft=m["n_fft"], hop_length=m["hop_length"],
        win_length=m["win_length"], n_mels=m["n_mels"],
        sample_rate=m["sample_rate"],
    )
    g_total = g_adv + 2.0 * g_fm + 2.0 * g_mel
    assert torch.isfinite(g_total), f"Generator loss non-finite: {g_total}"
    g_total.backward()
    opt_voc.step()
    ema_voc.update()

    # ── Discriminator step ────────────────────────────────────────────────
    disc.train()
    opt_disc.zero_grad()
    scores_real, scores_fake_d, _, _ = disc(wav_gt, wav_fake.detach())
    d_loss, _, _ = discriminator_loss_hinge(scores_real, scores_fake_d)
    assert torch.isfinite(d_loss), f"Discriminator loss non-finite: {d_loss}"
    d_loss.backward()
    opt_disc.step()


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="KazeFlow smoke test")
    p.add_argument("--device", default="auto",
                   help="Device to use: 'cpu', 'cuda', or 'auto' (default).")
    p.add_argument("--full", action="store_true",
                   help="Also test KazeFlowTrainer and KazeFlowPretrainer init.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    log.info("Using device: %s", device)

    cfg = copy.deepcopy(TINY_CONFIG)

    # ── Static / import tests ──────────────────────────────────────────────
    check("imports", test_imports)
    check("config load (load_config)", test_config_load)

    # ── Model tests ───────────────────────────────────────────────────────
    check("ConditionalFlowMatching  forward (CFM loss)",
          lambda: test_flow_matching_forward(cfg, device))
    check("ConditionalFlowMatching  sample (ODE inference)",
          lambda: test_flow_matching_sample(cfg, device))
    check("ChouwaGANGenerator       forward (mel→wav)",
          lambda: test_vocoder_forward(cfg, device))
    check("EMAGenerator             update + get_model",
          lambda: test_ema_generator(cfg, device))
    check("ChouwaGANDiscriminator   forward (real/fake)",
          lambda: test_discriminator_forward(cfg, device))

    # ── Loss tests ────────────────────────────────────────────────────────
    check("All loss functions execute",
          lambda: test_losses(cfg, device))

    # ── Dataset tests ────────────────────────────────────────────────────
    check("KazeFlowDataset + create_dataloader",
          lambda: test_dataset_and_dataloader(cfg))

    # ── Full training step ────────────────────────────────────────────────
    check("Full training step (CFM + GAN forward+backward)",
          lambda: test_full_training_step(cfg, device))

    # ── Trainer init (optional — slower, writes temp dirs) ─────────────
    if args.full:
        check("KazeFlowTrainer.__init__",
              lambda: test_trainer_init(cfg, device_str))
        check("KazeFlowPretrainer.__init__",
              lambda: test_pretrainer_init(cfg, device_str))
    else:
        log.info("%-55s %s  (pass --full to enable)", "KazeFlowTrainer init", SKIP)
        log.info("%-55s %s  (pass --full to enable)", "KazeFlowPretrainer init", SKIP)

    # ── Summary ──────────────────────────────────────────────────────────
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print()
    print(f"Results: {n_pass} passed, {n_fail} failed out of {len(_results)} tests.")
    if n_fail:
        print("\nFailed tests:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  {name}: {msg}")
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
