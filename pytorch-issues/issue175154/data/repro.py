"""K44 — pytorch/pytorch#175154: interpolate mode=nearest under torch.compile wrong result.

Inductor's lowering of `F.interpolate(mode='nearest')` produces a
different result than eager. Tests the interpolation kernel.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K44", title="interpolate nearest under inductor",
             issue=175154,
             expected="Inductor's interpolate(mode='nearest') diverges from eager.")

import torch
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(1, 4, 16, 16, device="cuda")

def fn(x):
    return F.interpolate(x, scale_factor=2.5, mode="nearest")

y_eager = fn(x)
fc = torch.compile(fn)
y_comp = fc(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"max abs diff: {(y_eager - y_comp).abs().max().item()}")
print(f"shapes eager {list(y_eager.shape)} compile {list(y_comp.shape)}")
