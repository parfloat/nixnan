"""K40 — pytorch/pytorch#182700: vllm cascade_attention regression on FLASH_ATTN.

vllm's repro depends on its test harness; we lift the underlying SDPA
flash-attn call into a minimal form so the FA kernel runs under nixnan.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K40", title="vllm cascade_attention regression (proxy SDPA FLASH)",
             issue=182700,
             expected="FA kernel regression caused extra token in vllm Fibonacci output; we just run the SDPA FLASH path.")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
B, H, N, D = 1, 8, 1024, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

try:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    print("torch:", torch.__version__)
    print(f"out shape: {out.shape}  nan={torch.isnan(out).any().item()} inf={torch.isinf(out).any().item()}")
except Exception as e:
    print("exception:", repr(e))
