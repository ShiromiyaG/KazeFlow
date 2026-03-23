"""
Fused SnakeBeta activation using Triton.

Replaces the multi-op SnakeBeta (cast → mul → sin → pow2 → div → add → cast)
with a single fused GPU kernel, reducing memory bandwidth ~3×.

Falls back to standard PyTorch ops when Triton is unavailable.
"""

import logging

import torch
import torch.nn as nn
from torch.nn import Parameter

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ─── Triton Kernels ──────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _snake_beta_fwd_kernel(
        X_ptr,
        Out_ptr,
        Alpha_ptr,
        Beta_ptr,
        N,
        C,
        T,
        alpha_logscale: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Forward: y = x + sin²(α·x) / (β + ε), computed in FP32."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        # Channel index from flat offset: for (B, C, T), c = (flat // T) % C
        c = (offs // T) % C

        # Per-channel parameters
        alpha = tl.load(Alpha_ptr + c, mask=mask).to(tl.float32)
        beta = tl.load(Beta_ptr + c, mask=mask).to(tl.float32)
        if alpha_logscale:
            alpha = tl.exp(alpha)
            beta = tl.exp(beta)
        alpha = tl.minimum(alpha, 100.0)
        beta = tl.minimum(beta, 100.0)

        # Load input in FP32
        x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # y = x + sin²(αx) / (β + ε)
        ax = x * alpha
        s = tl.sin(ax)
        y = x + s * s / (beta + EPS)

        # Store (auto-casts to output dtype)
        tl.store(Out_ptr + offs, y, mask=mask)

    @triton.jit
    def _snake_beta_bwd_x_kernel(
        GradOut_ptr,
        X_ptr,
        GradX_ptr,
        Alpha_ptr,
        Beta_ptr,
        N,
        C,
        T,
        alpha_logscale: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Backward for grad_x: dy/dx = 1 + 2α·sin(αx)·cos(αx) / (β + ε)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        c = (offs // T) % C

        alpha = tl.load(Alpha_ptr + c, mask=mask).to(tl.float32)
        beta = tl.load(Beta_ptr + c, mask=mask).to(tl.float32)
        if alpha_logscale:
            alpha = tl.exp(alpha)
            beta = tl.exp(beta)
        alpha = tl.minimum(alpha, 100.0)
        beta = tl.minimum(beta, 100.0)

        x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        go = tl.load(GradOut_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        ax = x * alpha
        sin_ax = tl.sin(ax)
        cos_ax = tl.cos(ax)
        denom = beta + EPS

        # dy/dx = 1 + 2α·sin(αx)·cos(αx) / (β + ε)
        grad_x = go * (1.0 + 2.0 * alpha * sin_ax * cos_ax / denom)

        tl.store(GradX_ptr + offs, grad_x, mask=mask)


# ─── Autograd Function ───────────────────────────────────────────────────────


class _SnakeBetaTriton(torch.autograd.Function):
    """Custom autograd: Triton forward + Triton grad_x + PyTorch param grads."""

    @staticmethod
    def forward(ctx, x, alpha, beta, alpha_logscale, eps):
        x = x.contiguous()
        B, C, T = x.shape
        N = x.numel()
        out = torch.empty_like(x)

        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        _snake_beta_fwd_kernel[grid](
            x, out, alpha, beta, N, C, T, alpha_logscale, eps, BLOCK=BLOCK
        )

        ctx.save_for_backward(x, alpha, beta)
        ctx.alpha_logscale = alpha_logscale
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta = ctx.saved_tensors
        B, C, T = x.shape
        N = x.numel()
        eps = ctx.eps

        # ── grad_x via Triton (fused, fast) ──────────────────────────────
        grad_output = grad_output.contiguous()
        grad_x = torch.empty_like(x)

        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)

        _snake_beta_bwd_x_kernel[grid](
            grad_output, x, grad_x, alpha, beta, N, C, T,
            ctx.alpha_logscale, eps, BLOCK=BLOCK,
        )

        # ── grad_alpha, grad_beta via PyTorch (reductions → small cost) ──
        x_f = x.float()
        go_f = grad_output.float()
        a = alpha.float().unsqueeze(0).unsqueeze(-1)  # (1, C, 1)
        b = beta.float().unsqueeze(0).unsqueeze(-1)

        if ctx.alpha_logscale:
            a_val = a.exp().clamp(max=100.0)
            b_val = b.exp().clamp(max=100.0)
        else:
            a_val = a.clamp(max=100.0)
            b_val = b.clamp(max=100.0)

        ax = a_val * x_f
        sin_ax = torch.sin(ax)
        cos_ax = torch.cos(ax)
        denom = b_val + eps

        # dy/dα = 2x·sin(αx)·cos(αx) / (β + ε), summed over (B, T)
        grad_alpha = (go_f * 2.0 * x_f * sin_ax * cos_ax / denom).sum(
            dim=(0, 2)
        )
        # dy/dβ = −sin²(αx) / (β + ε)², summed over (B, T)
        grad_beta = (go_f * (-sin_ax * sin_ax / (denom * denom))).sum(
            dim=(0, 2)
        )

        if ctx.alpha_logscale:
            # Chain rule for exp parametrization
            grad_alpha = grad_alpha * alpha.float().exp()
            grad_beta = grad_beta * beta.float().exp()

        return grad_x, grad_alpha, grad_beta, None, None


# ─── Fallback (pure PyTorch) ─────────────────────────────────────────────────


def _snake_beta_pytorch(x, alpha, beta, alpha_logscale, eps):
    """Standard SnakeBeta in PyTorch ops (FP32 safe)."""
    orig_dtype = x.dtype
    x = x.float()
    a = alpha.unsqueeze(0).unsqueeze(-1).float()
    b = beta.unsqueeze(0).unsqueeze(-1).float()
    if alpha_logscale:
        a = torch.exp(a)
        b = torch.exp(b)
    a = a.clamp(max=100.0)
    b = b.clamp(max=100.0)
    t = x.mul(a)
    t.sin_()
    t.pow_(2)
    t.div_(b + eps)
    return (x + t).to(orig_dtype)


# ─── Public API ──────────────────────────────────────────────────────────────


def snake_beta_forward(x, alpha, beta, alpha_logscale, eps=1e-9):
    """Dispatch to Triton kernel or PyTorch fallback."""
    if HAS_TRITON and x.is_cuda and x.dim() == 3:
        return _SnakeBetaTriton.apply(x, alpha, beta, alpha_logscale, eps)
    return _snake_beta_pytorch(x, alpha, beta, alpha_logscale, eps)
