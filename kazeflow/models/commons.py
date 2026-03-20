"""
Utility functions shared across KazeFlow modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import numpy as np


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Create boolean mask from sequence lengths. (B,) -> (B, T)"""
    if max_length is None:
        max_length = length.max().item()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    n = n_channels[0]
    x = input_a + input_b
    return torch.tanh(x[:, :n, :]) * torch.sigmoid(x[:, n:, :])


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
