# R-Spin: Efficient Speaker and Noise-invariant Representation Learning
# with Acoustic Pieces (NAACL 2024)
#
# Original code: https://github.com/vectominist/rspin
# License: MIT (Copyright (c) 2024 Heng-Jui Chang)
#
# WavLM backbone: https://github.com/microsoft/unilm/tree/master/wavlm
# License: MIT (Copyright (c) 2021 Microsoft)

from .model import RSpinWavlm

__all__ = ["RSpinWavlm"]
