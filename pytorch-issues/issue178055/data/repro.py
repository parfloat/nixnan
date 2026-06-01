"""K32 — pytorch/pytorch#178055: torch.compile produces Inf on fp32 boundary input.

Upstream observes eager finite vs compile Inf for the same Conv2d
call. We use synthetic large-magnitude input/weights so the Triton
output kernels run, then compare eager vs compile output.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K32", title="torch.compile Inf on fp32 boundary",
             issue=178055,
             expected="Compiled Conv2d produces Inf where eager stays finite.")

import torch
import torch.nn as nn

torch.manual_seed(0)
x = (torch.rand(1, 3, 32, 32) * 3.0e+38).cuda()
m = nn.Conv2d(3, 64, 3, padding=1, bias=True).eval().cuda()

with torch.no_grad():
    y_eager = m(x)
m_c = torch.compile(m, dynamic=False)
with torch.no_grad():
    y_comp = m_c(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"eager  : nan={torch.isnan(y_eager).any().item()} inf={torch.isinf(y_eager).any().item()}")
print(f"compile: nan={torch.isnan(y_comp).any().item()} inf={torch.isinf(y_comp).any().item()}")
