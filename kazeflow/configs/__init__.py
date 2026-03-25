"""
KazeFlow config loading with layered merging.

Configs are organized as:
  base.json           — Full reference config (48kHz finetune, all defaults)
  sr/<SR>.json        — Sample-rate-dependent overrides
  presets/pretrain.json    — Pretrain-specific overrides

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


def apply_vocoder_overlay(config: dict, vocoder_type: Optional[str]) -> dict:
    """Apply vocoder specific overlay to an existing config."""
    if vocoder_type is not None:
        config["model"]["vocoder_type"] = vocoder_type
        voc_path = _CONFIGS_DIR / "vocoder" / f"{vocoder_type}.json"
        if voc_path.exists():
            with open(voc_path) as f:
                voc_overlay = json.load(f)
            voc_overlay.pop("_doc", None)
            config["model"]["vocoder"] = _deep_merge(
                config["model"].get("vocoder", {}), voc_overlay
            )
        disc_path = _CONFIGS_DIR / "discriminator" / f"{vocoder_type}.json"
        if disc_path.exists():
            with open(disc_path) as f:
                disc_overlay = json.load(f)
            disc_overlay.pop("_doc", None)
            config["model"]["discriminator"] = _deep_merge(
                config["model"].get("discriminator", {}), disc_overlay
            )
    return config

def load_config(
    sample_rate: int = 48000,
    preset: Optional[str] = None,
    vocoder_type: Optional[str] = None,
) -> dict:
    """Load a layered config: base → SR overlay → preset overlay → vocoder overlay.

    Args:
        sample_rate: 32000, 40000, 44100, or 48000.
        preset: None (finetune) or "pretrain".
        vocoder_type: None (use base default) or a registered vocoder name.

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
    if preset == "pretrain":
        p = _CONFIGS_DIR / "presets" / "pretrain.json"
        if p.exists():
            with open(p) as f:
                config = _deep_merge(config, json.load(f))
    elif preset is not None:
        raise ValueError(f"Unknown preset '{preset}'. Available: pretrain")

    # 4) Vocoder type overlay — load per-type vocoder & discriminator configs
    return apply_vocoder_overlay(config, vocoder_type)
