"""K5 — pytorch/pytorch#173799: F.pdist(p=0) on inf input disagrees CPU vs CUDA.

Expected:
  CPU Result: tensor([nan, nan, nan, nan, nan, nan], dtype=torch.float64)
  GPU Result: tensor([2.,  2.,  2.,  2.,  2.,  2.], dtype=torch.float64)
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K5', title='F.pdist(p=0) on inf rows', issue=173799, expected='CUDA returns 2.0; CPU returns NaN (because inf-inf=NaN).')

import torch
import torch.nn.functional as F

x = torch.tensor(
    [[-float("inf"),  float("inf")],
     [ float("inf"),  float("inf")],
     [ float("inf"),  float("inf")],
     [ float("inf"),  float("inf")]],
    dtype=torch.float64,
)

y_cpu = F.pdist(x, p=0.0)
y_gpu = F.pdist(x.cuda(), p=0.0).cpu()
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("CPU :", y_cpu)
print("CUDA:", y_gpu)
print("cpu_nan :", torch.isnan(y_cpu).any().item())
print("cuda_nan:", torch.isnan(y_gpu).any().item())
print("divergence:", torch.isnan(y_cpu).any().item() != torch.isnan(y_gpu).any().item())
