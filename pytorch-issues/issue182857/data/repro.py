"""K38 — pytorch/pytorch#182857: torch.compile crashes on SDPA backward when head_dim is not a multiple of 16.

Use head_dim=15 to trigger the edge case under the FA kernel.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K38", title="SDPA backward head_dim%16 != 0",
             issue=182857,
             expected="torch.compile crashes on SDPA backward when head_dim is not a multiple of 16.")

import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, H, N, D = 2, 4, 64, 15  # D=15 is not a multiple of 16
q = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
k = torch.randn(B, H, N, D, device="cuda", requires_grad=True)
v = torch.randn(B, H, N, D, device="cuda", requires_grad=True)

def fn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

try:
    fc = torch.compile(fn)
    out = fc(q, k, v)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    print("torch:", torch.__version__)
    print(f"grads ok: q.nan={torch.isnan(q.grad).any().item()} v.nan={torch.isnan(v.grad).any().item()}")
except Exception as e:
    print("exception:", repr(e))
