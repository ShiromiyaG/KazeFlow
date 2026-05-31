"""
KazeFlow models — architecture factory.

Provides ``build_mel_model()`` to instantiate:
- ``RectifiedFlowMatching`` (RFM) — Rectified Flow (default, recommended)

All models share the same ``forward(x_1, x_mask, content, f0, g)`` training
interface and ``sample(content, f0, x_mask, g, ...)`` inference interface.
"""


def build_mel_model(architecture: str, **kwargs):
    """Build the mel generation model based on architecture choice.

    Args:
        architecture: ``"rfm"`` or ``"reflow"`` (aliases for RectifiedFlowMatching)
        **kwargs: Passed to the model constructor (from config ``flow_matching``).

    Returns:
        A model with ``.forward()`` for training loss and ``.sample()`` for inference.
    """
    from kazeflow.models.rectified_flow import RectifiedFlowMatching
    return RectifiedFlowMatching(**kwargs)
