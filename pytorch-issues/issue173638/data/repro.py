"""K3 — pytorch/pytorch#173638: torch.linalg.slogdet does not propagate NaN in CUDA.

Expected:
  logabsdet for A [CPU]  nan
  logabsdet for A [CUDA] 3.1354942321777344     (silently swallows the NaN row)
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K3', title='torch.linalg.slogdet swallows NaN row', issue=173638, expected='CUDA returns a finite logabsdet on a NaN-containing matrix; CPU returns NaN.')

import torch

A = torch.tensor(
    [[-1.0, 2.0, 3.0],
     [float("nan"), 5.0, 6.0],
     [7.0, 8.0, 10.0]],
    dtype=torch.float32,
)

s_cpu, l_cpu = torch.linalg.slogdet(A)
s_gpu, l_gpu = torch.linalg.slogdet(A.cuda())
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"logabsdet for A [CPU ]: {l_cpu.item()}")
print(f"logabsdet for A [CUDA]: {l_gpu.item()}")
print("cpu_is_nan :", torch.isnan(l_cpu).item())
print("cuda_is_nan:", torch.isnan(l_gpu).item())
print("divergence:", torch.isnan(l_cpu).item() != torch.isnan(l_gpu).item())
