"""K39 — pytorch/pytorch#181645: flex_attention(BACKEND='FLASH', return_lse=True) returns lse off by ln(2).

Only manifests inside torch.compile when the FLASH backend dispatches
to FA4. On older PyTorch/our local install the Triton backend (which
honours the base-2 convention) is used instead, so the bug may not
reproduce; the kernel still runs.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K39", title="flex_attention FLASH return_lse off by ln(2)",
             issue=181645,
             expected="With FLASH backend, returned lse is multiplied by ln(2) extra.")

import math
import torch
try:
    from torch.nn.attention.flex_attention import flex_attention
except Exception as e:
    print("flex_attention not available:", repr(e))
    raise SystemExit(0)

torch.manual_seed(0)
N, D = 256, 32
q = torch.randn(1, 1, N, D, device="cuda", dtype=torch.float16)
k = torch.randn(1, 1, N, D, device="cuda", dtype=torch.float16)
v = torch.randn(1, 1, N, D, device="cuda", dtype=torch.float16)

ref = torch.logsumexp(q.squeeze().float() @ k.squeeze().float().T, dim=-1)
try:
    fc = torch.compile(lambda q, k, v: flex_attention(
        q, k, v, kernel_options={"BACKEND": "TRITON"}, scale=1.0, return_lse=True))
    _, lse_t = fc(q, k, v)
    torch.cuda.synchronize()
    print("torch:", torch.__version__)
    print(f"max |lse_triton - ref|: {(lse_t.squeeze().float() - ref).abs().max().item()}")
except Exception as e:
    print("exception:", repr(e))
