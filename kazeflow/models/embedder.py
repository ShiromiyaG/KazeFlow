"""
KazeFlow — Unified Content Embedder Loader.

Supports:
  - spin_v2  : HuBERT-based SPIN v2 (dr87/spinv2_rvc)  →  768-dim
  - rspin    : WavLM-based R-Spin with VQ bottleneck     →  codebook_dim (256)

Output: (B, T_frames, output_dim) at 50 Hz from 16 kHz input.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("kazeflow.embedder")

# Mapping from embedder name → output dimension (for config auto-setup)
EMBEDDER_DIMS = {
    "spin_v2": 768,
    "rspin": 256,   # pred_head bottleneck: 768 → codebook_dim
}

_ROOT = Path(__file__).parent.parent  # kazeflow/models/ → kazeflow/

_SPIN_V2_DIR = _ROOT / "models" / "pretrained" / "embedders" / "spin_v2"
_RSPIN_DIR = _ROOT / "models" / "pretrained" / "embedders" / "rspin"


class _SpinV2Wrapper(nn.Module):
    """Thin wrapper around HubertModel to provide a unified interface."""

    output_dim: int = 768

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) at 16 kHz → features: (B, T_frames, 768)"""
        return self.model(wav).last_hidden_state


class _RSpinWrapper(nn.Module):
    """Continuous bottleneck wrapper around RSpinWavlm.

    Uses the ``pred_head`` projection (768 → codebook_dim) + L2-norm
    as a dimension bottleneck that strips residual timbre while
    preserving phonetic content.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output_dim: int = model.cfg["codebook_dim"]

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) at 16 kHz → features: (B, T_frames, codebook_dim)"""
        feat_list, _pad_mask, _codes = self.model(wav)
        return F.normalize(self.model.pred_head(feat_list[-1]), dim=-1)  # (B, T, codebook_dim)


def load_content_embedder(
    name: str = "spin_v2",
    device: str = "cpu",
    spin_source: Optional[str] = None,
    rspin_checkpoint: Optional[str] = None,
) -> nn.Module:
    """
    Load a content embedder model by name.

    Args:
        name:             "spin_v2" or "rspin"
        device:           target device
        spin_source:      optional override path/HF hub ID for SPIN v2
        rspin_checkpoint: optional override path for the RSPIN .pt file

    Returns:
        nn.Module with forward(wav) → (B, T_frames, output_dim)
            output_dim: 768 for spin_v2, 256 for rspin
    """
    if name == "spin_v2":
        from transformers import HubertModel

        if spin_source is not None:
            source = spin_source
        elif _SPIN_V2_DIR.exists():
            source = str(_SPIN_V2_DIR)
        else:
            source = "dr87/spinv2_rvc"
            logger.warning(
                "Local SPIN v2 not found at %s — falling back to HF Hub '%s'",
                _SPIN_V2_DIR, source,
            )

        model = HubertModel.from_pretrained(source)
        wrapper = _SpinV2Wrapper(model)

    elif name == "rspin":
        import warnings
        from kazeflow.models.rspin import RSpinWavlm

        if rspin_checkpoint is not None:
            ckpt_path = Path(rspin_checkpoint)
        else:
            # Auto-detect the first .pt file in the rspin directory
            ckpt_path = None
            if _RSPIN_DIR.exists():
                pts = sorted(_RSPIN_DIR.glob("*.pt"))
                if pts:
                    ckpt_path = pts[0]

        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(
                f"RSPIN checkpoint not found at {ckpt_path or _RSPIN_DIR}. "
                "Place a wavlm_rspin_*.pt file in "
                f"{_RSPIN_DIR} or pass rspin_checkpoint=."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)
            model = RSpinWavlm.load_from_checkpoint(str(ckpt_path))
        wrapper = _RSpinWrapper(model)

        logger.info("RSPIN checkpoint: %s", ckpt_path.name)

    else:
        raise ValueError(
            f"Unknown content embedder '{name}'. Choose 'spin_v2' or 'rspin'."
        )

    wrapper = wrapper.to(device).float().eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    logger.info("Loaded content embedder: %s (output_dim=%d)", name, wrapper.output_dim)
    return wrapper
