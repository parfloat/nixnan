# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math

import torch
import torch.nn.functional as F
import cuda.tile as ct

from test.kernels.layer_norm import (
    layer_norm_fwd, layer_norm_bwd_dx_partial_dwdb, layer_norm_bwd_dwdb
)


# --- cuTile LayerNorm Wrapper ------------------------------------------------------

class CuTileLayerNorm(torch.autograd.Function):
    """
    A PyTorch Autograd Function wrapper for the cuTile LayerNorm kernel.
    This class manages the forward and backward passes, bridging PyTorch tensors
    with the cuTile kernel launches.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        """
        Forward pass for LayerNorm.

        Args:
            ctx: Context object to save tensors for backward pass.
            input: Input tensor (*, ..., N).
            weight: Scale parameter (N,).
            bias: Shift parameter (N,).
            eps: Epsilon for numerical stability.

        Returns:
            Output tensor with the same shape as input.
        """
        # Flatten input to (M, N)
        x = input.reshape(-1, input.shape[-1])
        y = torch.empty_like(x)
        M, _ = x.shape

        # Allocate temporary buffers for mean and reciprocal standard deviation
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        TILE_N = 1024
        # Launch the forward kernel with a 1D grid (M blocks)
        ct.launch(torch.cuda.current_stream(), (M,), layer_norm_fwd,
                  (x, weight, bias, y, mean, rstd, eps, TILE_N))

        # Save tensors needed for the backward pass
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.TILE_N = TILE_N

        return y.reshape(*input.shape)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for LayerNorm.

        Computes gradients for input, weight, and bias using two kernels:
        1. Computes dX and partial reductions for dW and dB.
        2. Performs the final reduction for dW and dB.

        Args:
            ctx: Context object containing saved tensors.
            grad_output: Gradient tensor of loss w.r.t. output (*, ..., N).

        Returns:
            Gradients for input, weight, bias, and None for eps.
        """
        x, weight, bias, mean, rstd = ctx.saved_tensors
        TILE_N = ctx.TILE_N
        M, N = x.shape
        GROUP_SIZE_M = 64

        # Flatten gradient output to (M, N)
        dy = grad_output.reshape(-1, grad_output.shape[-1])
        dx = torch.empty_like(dy)

        # Allocate buffers for partial gradients and synchronization locks
        dw = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=weight.device)
        db = torch.zeros((GROUP_SIZE_M, N), dtype=torch.float32, device=bias.device)
        locks = torch.zeros(GROUP_SIZE_M, dtype=torch.int32, device=weight.device)

        # Launch the first backward kernel to compute dX and partial dW/dB
        ct.launch(torch.cuda.current_stream(), (M,), layer_norm_bwd_dx_partial_dwdb,
                  (dx, dy, dw, db, x, weight, mean, rstd, locks, TILE_N))

        final_dw = torch.empty((N,), dtype=weight.dtype, device=weight.device)
        final_db = torch.empty((N,), dtype=bias.dtype, device=bias.device)
        TILE_M = 32

        # Launch the second backward kernel to reduce partial dW/dB
        ct.launch(torch.cuda.current_stream(), (math.ceil(N / TILE_N),), layer_norm_bwd_dwdb,
                  (dw, db, final_dw, final_db, TILE_M, TILE_N))

        return dx.reshape(*grad_output.shape), final_dw, final_db, None


def cutile_layer_norm(x, weight, bias, eps):
    return CuTileLayerNorm.apply(x, weight, bias, eps)


# --- Torch Reference Implementation -----------------------------------------

def torch_layer_norm(x, weight, bias, eps):
    return F.layer_norm(x, weight.shape, weight, bias, eps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the cuTile LayerNorm output against a torch reference.",
    )
    args = parser.parse_args()

    print("--- Running cuTile LayerNorm Forward/Backward Sample ---")

    shape = (1024, 2048)
    dtype = torch.bfloat16
    weight = torch.randn(shape[-1], dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.randn(shape[-1], dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(shape, dtype=dtype, device='cuda')
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    eps = 1e-5

    print(f"Input shape: {shape}, dtype: {dtype}, eps: {eps}")

    atol, rtol = 1e-2, 1e-2

    print("\n--- Executing cuTile LayerNorm Forward ---")
    y = cutile_layer_norm(x, weight, bias, eps)\

    print("\n--- Executing cuTile LayerNorm Backward ---")
    y.backward(dy, retain_graph=True)
    dx, dw, db = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None

    if args.correctness_check:
        print("\n--- Running correctness check against torch reference ---")
        y_ref = torch_layer_norm(x, weight, bias, eps)
        torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)

        y_ref.backward(dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
        torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(dw, dw_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(db, db_ref, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled (use --correctness-check to enable)")

    print("\n--- cuTile LayerNorm Sample complete ---")
