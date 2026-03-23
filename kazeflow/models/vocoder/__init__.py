"""
KazeFlow Vocoder package.

Each vocoder lives in its own module (e.g. chouwa_gan.py).
To add a new vocoder:
  1. Create a new file in this directory (e.g. my_vocoder.py)
  2. Implement a generator class with the interface:
       forward(mel, f0, g=None) → waveform  (B, 1, T_audio)
       remove_weight_norm() → None
       register_output_grad_clip(max_norm: float) → None  (optional)
       get_compilable_module() → nn.Module  (optional)
  3. Register it in VOCODER_REGISTRY below
"""

from kazeflow.models.vocoder.chouwa_gan import (  # noqa: F401
    ChouwaGANGenerator,
    EMAGenerator,
)

# ── Registry ─────────────────────────────────────────────────────────────────
# Maps vocoder_type string → generator class.
# Each class must accept (sr=int, **vocoder_config_kwargs).
VOCODER_REGISTRY: dict[str, type] = {
    "chouwa_gan": ChouwaGANGenerator,
}


def build_vocoder(vocoder_type: str, sr: int, **kwargs):
    """Instantiate a vocoder generator by type name.

    Args:
        vocoder_type: Key in VOCODER_REGISTRY (e.g. "chouwa_gan").
        sr: Sample rate.
        **kwargs: Vocoder-specific config (forwarded to constructor).

    Returns:
        nn.Module with forward(mel, f0, g=None) → waveform interface.
    """
    if vocoder_type not in VOCODER_REGISTRY:
        available = ", ".join(sorted(VOCODER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown vocoder_type '{vocoder_type}'. Available: {available}"
        )
    return VOCODER_REGISTRY[vocoder_type](sr=sr, **kwargs)
