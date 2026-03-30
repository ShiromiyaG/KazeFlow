"""
KazeFlow models — architecture factory.

Provides ``build_mel_model()`` to instantiate either:
- ``ConditionalFlowMatching`` (CFM) — ODE-based generative model
- ``DirectMelPredictor`` — single-pass regression (simpler, faster)

Both share the same ``forward(x_1, x_mask, content, f0, g)`` training
interface and ``sample(content, f0, x_mask, g, ...)`` inference interface.
"""


def build_mel_model(architecture: str, **kwargs):
    """Build the mel generation model based on architecture choice.

    Args:
        architecture: ``"cfm"`` or ``"direct_mel"``
        **kwargs: Passed to the model constructor (from config ``flow_matching``).

    Returns:
        A model with ``.forward()`` for training loss and ``.sample()`` for inference.
    """
    if architecture == "direct_mel":
        from kazeflow.models.mel_predictor import DirectMelPredictor
        return DirectMelPredictor(**kwargs)
    else:
        from kazeflow.models.flow_matching import ConditionalFlowMatching
        return ConditionalFlowMatching(**kwargs)
