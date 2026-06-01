"""K25 — pytorch/pytorch#178251: SDPA backward returns NaN in 0th output with attn_mask.

Upstream observed NaN on `ScaledDotProductEfficientAttentionBackward0`
when using `attn_mask` under the EFFICIENT_ATTENTION backend. We
build a small Q/K/V plus a mask and run forward+backward to drive
the SDPA backward path.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K25", title="SDPA backward NaN with attn_mask",
             issue=178251,
             expected="SDPA backward returns NaN gradients on q/k under EFFICIENT_ATTENTION + attn_mask.")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
B, H, N, D = 2, 4, 64, 32
q = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
k = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
v = torch.randn(B, H, N, D, device="cuda", requires_grad=True)

# Float mask with -inf entries — common cause of SDPA backward NaN
attn_mask = torch.zeros(B, 1, N, N, device="cuda")
attn_mask[:, :, :, N//2:] = float("-inf")

try:
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    print("torch:", torch.__version__)
    print(f"q.grad nan={torch.isnan(q.grad).any().item()}  inf={torch.isinf(q.grad).any().item()}")
    print(f"k.grad nan={torch.isnan(k.grad).any().item()}  inf={torch.isinf(k.grad).any().item()}")
    print(f"v.grad nan={torch.isnan(v.grad).any().item()}  inf={torch.isinf(v.grad).any().item()}")
except Exception as e:
    print("exception:", repr(e))
