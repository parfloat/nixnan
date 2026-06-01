"""K8 — pytorch/pytorch#165537: torch.linspace(0, +inf, 3, fp16) returns all NaN on CUDA.

NumPy returns [nan, inf, inf]; PyTorch returns [nan, nan, nan].
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K8', title='torch.linspace(0,inf,3,fp16)', issue=165537, expected='Returns all NaN; NumPy returns [nan, inf, inf].')

import torch

start, end, num = 0.0, float("inf"), 3
cpu = torch.linspace(start, end, num, dtype=torch.float16, device="cpu")
gpu = torch.linspace(start, end, num, dtype=torch.float16, device="cuda").cpu()
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("CPU :", cpu)
print("CUDA:", gpu)
