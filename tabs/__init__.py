"""KazeFlow UI tabs — shared helpers."""

import logging
import sys

logger = logging.getLogger("kazeflow.ui")


def _gpu_caps() -> dict:
    """Detect GPU precision and compilation support.

    Returns a dict with:
        has_tf32:  bool — GPU supports TF32 (Ampere+, compute >= 8.0)
        has_bf16:  bool — GPU supports BF16 (Ampere+, compute >= 8.0)
        has_compile: bool — torch.compile + Triton available (Volta+, compute >= 7.0)
    """
    has_tf32 = False
    has_bf16 = False
    has_compile = False

    try:
        import torch

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            has_tf32 = major >= 8
            has_bf16 = major >= 8
            # torch.compile uses Triton backend which requires SM >= 7.0 (Volta+)
            has_compile = hasattr(torch, "compile") and major >= 7
        else:
            has_compile = hasattr(torch, "compile")
    except Exception:
        logger.debug("GPU capability detection failed", exc_info=True)

    # Triton (torch.compile backend) is not supported on Windows
    if sys.platform == "win32":
        has_compile = False

    return {"has_tf32": has_tf32, "has_bf16": has_bf16, "has_compile": has_compile}


def get_precision_choices(caps: dict | None = None) -> tuple[list[str], str]:
    """Return (choices, default) for the precision radio based on GPU caps."""
    if caps is None:
        caps = _gpu_caps()

    all_choices = ["fp32", "fp32_fp16", "tf32", "tf32_fp16", "tf32_bf16"]
    filtered = []
    for c in all_choices:
        if "tf32" in c and not caps["has_tf32"]:
            continue
        if "bf16" in c and not caps["has_bf16"]:
            continue
        filtered.append(c)

    if not filtered:
        filtered = ["fp32"]

    default = "tf32_fp16" if "tf32_fp16" in filtered else filtered[-1]
    return filtered, default
