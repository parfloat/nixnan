# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math

import torch
import torch.nn.functional as F
import cuda.tile as ct

from test.kernels.fused_moe import fused_moe_kernel, moe_align_tile_size_torch, silu_and_mul_kernel


# --- cuTile MoE Wrapper ------------------------------------------------------
def cutile_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> torch.Tensor:
    """
    Executes a Mixture-of-Experts (MoE) forward pass using the fused cuTile kernel.

    Args:
        hidden_states: Token activations, shape (num_tokens, hidden_size)
        w1: Expert gate+up projection weights,
            shape (num_experts, intermediate_size * 2, hidden_size)
        w2: Expert down projection weights,
            shape (num_experts, hidden_size, intermediate_size)
        topk_weights: Router weights per token, shape (num_tokens, topk)
        topk_ids: Expert indices per token, shape (num_tokens, topk)
        tile_m/n/k: Tile sizes for cuTile kernel launch

    Returns:
        Tensor with the same shape/dtype as `hidden_states`.
    """
    out_dtype = hidden_states.dtype
    device = hidden_states.device

    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, intermediate_size = w2.shape
    _, topk = topk_ids.shape

    if w1.shape[1] != intermediate_size * 2:
        raise ValueError("w1 must have 2 * intermediate_size rows (gate + up projection)")

    intermediate_cache1 = torch.zeros(
        (num_tokens, topk, intermediate_size * 2),
        device=device,
        dtype=out_dtype,
    )
    intermediate_cache2 = torch.zeros(
        (num_tokens * topk, intermediate_size),
        device=device,
        dtype=out_dtype,
    )
    intermediate_cache3 = torch.zeros(
        (num_tokens, topk, hidden_size),
        device=device,
        dtype=out_dtype,
    )

    sorted_token_ids, sorted_expert_ids = moe_align_tile_size_torch(
        topk_ids,
        tile_m,
        num_experts,
    )

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        mul_routed_weight=False,
        num_token_replicas=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    invoke_silu_and_mul_kernel(
        intermediate_cache1.view(-1, intermediate_cache1.shape[-1]),
        intermediate_cache2,
    )

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        mul_routed_weight=True,
        num_token_replicas=1,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    return torch.sum(intermediate_cache3, dim=1)


# --- Torch Reference Implementation -----------------------------------------
def torch_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Naive torch implementation of MoE for correctness checks.
    """
    gate_proj, up_proj = w1.chunk(2, dim=1)
    down_proj = w2

    num_experts = w1.shape[0]
    final_hidden_states = torch.zeros_like(hidden_states)

    expert_mask = F.one_hot(topk_ids, num_classes=num_experts).permute(2, 1, 0)
    expert_usage = expert_mask.sum(dim=(-1, -2)) > 0
    active_expert_ids = expert_usage.nonzero().squeeze(-1)

    for expert_id in active_expert_ids:
        expert_gate = gate_proj[expert_id]
        expert_up = up_proj[expert_id]
        expert_down = down_proj[expert_id]

        matched_ks, matched_token_ids = torch.where(expert_mask[expert_id])
        matched_tokens = hidden_states[matched_token_ids]

        gate_output = matched_tokens @ expert_gate.T
        up_output = matched_tokens @ expert_up.T
        swiglu_output = F.silu(gate_output) * up_output
        expert_output = swiglu_output @ expert_down.T

        routing_weights = topk_weights[matched_token_ids, matched_ks]
        weighted_output = expert_output * routing_weights.unsqueeze(-1)

        final_hidden_states.index_add_(
            0,
            matched_token_ids,
            weighted_output.to(hidden_states.dtype),
        )

    return final_hidden_states


# --- Helper Utilities -------------------------------------------------------
def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    mul_routed_weight: bool,
    num_token_replicas: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    m = sorted_token_ids.shape[0]
    n = B.shape[1]

    grid = (math.ceil(m / tile_m) * math.ceil(n / tile_n),)
    topk_weights = topk_weights.view(-1)
    C = C.view(-1, C.shape[2])

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_moe_kernel,
        (
            A,
            B,
            C,
            topk_weights,
            sorted_token_ids,
            sorted_expert_ids,
            num_token_replicas,
            mul_routed_weight,
            tile_m,
            tile_n,
            tile_k,
        ),
    )


def invoke_silu_and_mul_kernel(
    AB: torch.Tensor,
    C: torch.Tensor
):
    A, B = AB.chunk(2, dim=-1)
    ct.launch(
        torch.cuda.current_stream(),
        (AB.shape[0],),
        silu_and_mul_kernel,
        (
            A,
            B,
            C,
            next_power_of_2(C.shape[-1])
        )
    )


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the cuTile MoE output against a torch reference.",
    )
    args = parser.parse_args()

    print("--- Running cuTile Mixture-of-Experts (MoE) Sample ---")

    num_tokens = 48
    hidden_size = 512
    num_experts = 64
    intermediate_size = 1024
    topk = 8
    dtype = torch.bfloat16

    device = "cuda"
    print(
        f"Tokens: {num_tokens}, Hidden: {hidden_size}, "
        f"Experts: {num_experts}, Intermediate: {intermediate_size}, "
        f"TopK: {topk}, dtype: {dtype}"
    )

    hidden_states = torch.empty(
        num_tokens, hidden_size, device=device, dtype=dtype
    ).normal_(0, 0.5)
    w1 = torch.empty(
        num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype
    ).normal_(0, 0.1)
    w2 = torch.empty(
        num_experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ).normal_(0, 0.1)

    # Unique expert IDs for each token (no repeating elements per row)
    topk_ids = torch.stack([
        torch.randperm(num_experts, device=device)[:topk]
        for _ in range(num_tokens)
    ])
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device=device), dim=-1
    ).to(dtype)

    print("\n--- Executing cuTile MoE ---")
    output_cutile = cutile_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                               tile_m=128, tile_n=128, tile_k=64)
    print(f"cuTile MoE output shape: {output_cutile.shape}, "
          "dtype: {output_cutile.dtype}")

    if args.correctness_check:
        print("\n--- Running correctness check against torch reference ---")
        ref_output = torch_moe(hidden_states, w1, w2, topk_weights, topk_ids)
        torch.testing.assert_close(output_cutile, ref_output, rtol=1e-1, atol=1e-1)
        print("Correctness check passed")
    else:
        print("Correctness check disabled (use --correctness-check to enable)")

    print("\n--- cuTile Mixture-of-Experts (MoE) Sample complete ---")
