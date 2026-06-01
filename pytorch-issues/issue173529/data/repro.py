"""K6 — pytorch/pytorch#173529: nn.Conv2d produces NaN on CUDA for boundary fp32 input.

Needs input.pt and weight_bias.pt from the issue's repro.zip
(github.com/user-attachments/files/24889013/repro.zip).

Expected:
  [CUDA] Out: nan=True, inf=True
  [CPU ] Out: nan=False, inf=False
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K6', title='nn.Conv2d near FLT_MAX', issue=173529, expected='CUDA produces NaN/Inf; CPU stays finite.')

from pathlib import Path
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
inp = torch.load(BASE / "input.pt", map_location="cpu", weights_only=True)
weights = torch.load(BASE / "weight_bias.pt", map_location="cpu", weights_only=True)

conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
conv.weight.data.copy_(weights["weight"])
conv.bias.data.copy_(weights["bias"])
conv.eval()

input_diff = (inp.cuda().cpu() - inp.cpu()).abs().max().item()
print("torch:", torch.__version__)
print(f"Input Consistency: {input_diff:.6e}")
print(f"Input Range: min={inp.min().item():.2e}, max={inp.max().item():.2e}, nan={torch.isnan(inp).any().item()}")

conv.to("cuda")
with torch.no_grad():
    out_cuda = conv(inp.cuda()).cpu()
print(f"[CUDA] nan={torch.isnan(out_cuda).any().item()}, inf={torch.isinf(out_cuda).any().item()}, "
      f"min={out_cuda.min().item():.2e}, max={out_cuda.max().item():.2e}")

conv.to("cpu")
with torch.no_grad():
    out_cpu = conv(inp.cpu())
print(f"[CPU ] nan={torch.isnan(out_cpu).any().item()}, inf={torch.isinf(out_cpu).any().item()}, "
      f"min={out_cpu.min().item():.2e}, max={out_cpu.max().item():.2e}")

cuda_bad = torch.isnan(out_cuda).any().item() or torch.isinf(out_cuda).any().item()
cpu_bad  = torch.isnan(out_cpu ).any().item() or torch.isinf(out_cpu ).any().item()
print("divergence:", cuda_bad != cpu_bad)
