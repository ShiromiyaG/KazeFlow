"""
KazeFlow config loading with layered merging.

Configs are organized as:
  base.json           — Full reference config (48kHz finetune, all defaults)
  sr/<SR>.json        — Sample-rate-dependent overrides
  presets/pretrain.json    — Pretrain-specific overrides
  presets/pretrain_v2.json — AFM v2 additions (layered on pretrain)

load_config() deep-merges: base → SR overlay → preset overlay
and returns a complete, ready-to-use config dict.
"""

import copy
import json
from pathlib import Path
from typing import Optional

_CONFIGS_DIR = Path(__file__).parent

_SR_MAP = {
    32000: "32k",
    40000: "40k",
    44100: "44k",
    48000: "48k",
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (overlay wins). Lists are replaced."""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    sample_rate: int = 48000,
    preset: Optional[str] = None,
) -> dict:
    """Load a layered config: base → SR overlay → preset overlay.

    Args:
        sample_rate: 32000, 40000, 44100, or 48000.
        preset: None (finetune), "pretrain", or "pretrain_v2".

    Returns:
        Full materialized config dict ready to use.
    """
    # 1) Base
    with open(_CONFIGS_DIR / "base.json") as f:
        config = json.load(f)

    # 2) SR overlay
    sr_key = _SR_MAP.get(sample_rate)
    if sr_key is None:
        raise ValueError(
            f"Unsupported sample_rate {sample_rate}. "
            f"Available: {sorted(_SR_MAP.keys())}"
        )
    sr_path = _CONFIGS_DIR / "sr" / f"{sr_key}.json"
    if sr_path.exists():
        with open(sr_path) as f:
            sr_overlay = json.load(f)
        if sr_overlay:
            config = _deep_merge(config, sr_overlay)

    # 3) Preset overlay
    if preset == "pretrain_v2":
        # v2 layers: pretrain base first, then v2 additions
        for name in ("pretrain", "pretrain_v2"):
            p = _CONFIGS_DIR / "presets" / f"{name}.json"
            if p.exists():
                with open(p) as f:
                    config = _deep_merge(config, json.load(f))
    elif preset == "pretrain":
        p = _CONFIGS_DIR / "presets" / "pretrain.json"
        if p.exists():
            with open(p) as f:
                config = _deep_merge(config, json.load(f))
    elif preset is not None:
        raise ValueError(f"Unknown preset '{preset}'. Available: pretrain, pretrain_v2")

    return config
