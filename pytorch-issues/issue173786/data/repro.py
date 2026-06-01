"""K4 — pytorch/pytorch#173786: torch.linalg.cholesky_ex silent NaN on CUDA for inf input.

Expected:
  CPU  | L: inf  info: 0
  CUDA | L: nan  info: 0      (silent failure: NaN with success status)
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K4', title='torch.linalg.cholesky_ex on [[inf]]', issue=173786, expected='CUDA returns silent NaN with info=0; CPU returns inf with info=0.')

import torch

A_cpu = torch.tensor([[float("inf")]], dtype=torch.float64)
L_cpu, info_cpu = torch.linalg.cholesky_ex(A_cpu, check_errors=False)

A_gpu = A_cpu.cuda()
L_gpu, info_gpu = torch.linalg.cholesky_ex(A_gpu, check_errors=False)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"CPU  | L: {L_cpu.item()}  info: {info_cpu.item()}")
print(f"CUDA | L: {L_gpu.item()}  info: {info_gpu.item()}")
print("cpu_nan :", torch.isnan(L_cpu).any().item())
print("cuda_nan:", torch.isnan(L_gpu).any().item())
print("divergence:", torch.isnan(L_cpu).any().item() != torch.isnan(L_gpu).any().item())
