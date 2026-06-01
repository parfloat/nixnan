"""K45 — pytorch/pytorch#178134: torch.compile Conv2d fp32 drift vs eager.

Smaller-magnitude sibling of K32 (#178055). Inductor's Conv2d
lowering drifts in fp32 even on inputs not near FLT_MAX.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K45", title="torch.compile Conv2d fp32 drift",
             issue=178134,
             expected="Inductor Conv2d drift vs eager at large but non-overflow inputs.")

import torch
import torch.nn as nn

torch.manual_seed(0)
x = torch.randn(1, 3, 64, 64, device="cuda") * 1e+15
m = nn.Conv2d(3, 32, 3, padding=1, bias=True).eval().cuda()

with torch.no_grad():
    y_e = m(x)
mc = torch.compile(m, dynamic=False)
with torch.no_grad():
    y_c = mc(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"max abs diff: {(y_e - y_c).abs().max().item():.3e}")
print(f"max rel diff: {((y_e - y_c).abs() / (y_e.abs() + 1e-30)).max().item():.3e}")
print(f"eager nan/inf: {torch.isnan(y_e).any().item()}/{torch.isinf(y_e).any().item()}  "
      f"compile nan/inf: {torch.isnan(y_c).any().item()}/{torch.isinf(y_c).any().item()}")
