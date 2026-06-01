"""K10 — pytorch/pytorch#179784: torch.xlogy(0, 0) returns NaN on CUDA.

Mathematically lim x*log(x) at x->0 is 0, but PyTorch evaluates 0 * -inf = NaN.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K10', title='torch.xlogy(0,0)', issue=179784, expected='Returns NaN; SciPy / common ML convention defines it as 0.')

import torch

v = torch.tensor([0.0, 0.5, 1.0, 2.0])
cpu = torch.xlogy(v, v)
gpu = torch.xlogy(v.cuda(), v.cuda()).cpu()
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("CPU :", cpu)
print("CUDA:", gpu)
print("any_nan:", torch.isnan(gpu).any().item())
