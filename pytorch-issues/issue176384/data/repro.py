"""K9 — pytorch/pytorch#176384: torch.erfinv on out-of-domain input.

erfinv's mathematical domain is (-1, 1); CUDA returns NaN for -1.47, etc.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K9', title='torch.erfinv on out-of-domain values', issue=176384, expected='PyTorch returns NaN for input < -1; PaddlePaddle returns -inf.')

import torch

x = torch.tensor([0.7778, -0.9117, 0.4359, -1.4654, -0.7388], dtype=torch.float32)
cpu = torch.erfinv(x)
gpu = torch.erfinv(x.cuda()).cpu()
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("CPU :", cpu)
print("CUDA:", gpu)
print("any_nan:", torch.isnan(gpu).any().item())
