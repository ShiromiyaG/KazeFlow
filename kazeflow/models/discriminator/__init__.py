"""
KazeFlow Discriminator package.

Each discriminator lives in its own module (e.g. chouwa_gan.py).
To add a new discriminator:
  1. Create a new file in this directory (e.g. my_disc.py)
  2. Implement a discriminator class with the interface:
       forward(y, y_hat, compute_fmaps=True)
         → (y_d_rs, y_d_gs, fmap_rs, fmap_gs)
       where y_d_* are lists of score tensors and fmap_* are lists of
       feature map lists.
  3. Register it in DISCRIMINATOR_REGISTRY below
"""

from kazeflow.models.discriminator.chouwa_gan import (  # noqa: F401
    ChouwaGANDiscriminator,
)

# ── Registry ─────────────────────────────────────────────────────────────────
# Maps discriminator_type string → discriminator class.
# Each class must accept (sample_rate=int, **disc_config_kwargs).
DISCRIMINATOR_REGISTRY: dict[str, type] = {
    "chouwa_gan": ChouwaGANDiscriminator,
}


def build_discriminator(discriminator_type: str, sample_rate: int, **kwargs):
    """Instantiate a discriminator by type name.

    Args:
        discriminator_type: Key in DISCRIMINATOR_REGISTRY (e.g. "chouwa_gan").
        sample_rate: Audio sample rate.
        **kwargs: Discriminator-specific config (forwarded to constructor).

    Returns:
        nn.Module with forward(y, y_hat, compute_fmaps) interface.
    """
    if discriminator_type not in DISCRIMINATOR_REGISTRY:
        available = ", ".join(sorted(DISCRIMINATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown discriminator_type '{discriminator_type}'. "
            f"Available: {available}"
        )
    return DISCRIMINATOR_REGISTRY[discriminator_type](
        sample_rate=sample_rate, **kwargs
    )
