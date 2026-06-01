"""K11 — pytorch/pytorch#173894: F.softplus with very large beta overflows.

exp(beta*x) overflows when beta=1e30 — produces inf where the answer should be x.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K11', title='F.softplus with beta=1e30', issue=173894, expected='Overflows to inf for positive inputs because exp(beta*x) overflows.')

import torch
import torch.nn.functional as F

x = torch.tensor([0.5, -1.0, 2.0])
cpu = F.softplus(x,         beta=1e30, threshold=1e30)
gpu = F.softplus(x.cuda(),  beta=1e30, threshold=1e30).cpu()
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("CPU :", cpu)
print("CUDA:", gpu)
print("any_inf cuda:", torch.isinf(gpu).any().item())
