"""K33 — pytorch/pytorch#178084: torch.compile introduces NaN in LayerNorm.

Eager LayerNorm stays finite on fp32 boundary inputs; the compiled
path produces NaN. Tests the Inductor-emitted normalisation kernel.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K33", title="torch.compile LayerNorm NaN",
             issue=178084,
             expected="Compiled LayerNorm produces NaN on fp32 boundary input where eager stays finite.")

import torch
import torch.nn as nn

torch.manual_seed(0)
x = (torch.rand(8, 128).cuda() * 1e+37) + 1e+37
m = nn.LayerNorm(128).cuda().eval()

with torch.no_grad():
    y_eager = m(x)
m_c = torch.compile(m, dynamic=False)
with torch.no_grad():
    y_comp = m_c(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"eager  : nan={torch.isnan(y_eager).any().item()} inf={torch.isinf(y_eager).any().item()}")
print(f"compile: nan={torch.isnan(y_comp).any().item()} inf={torch.isinf(y_comp).any().item()}")
