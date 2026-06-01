"""K34 — pytorch/pytorch#173793: LayerNorm under torch.compile produces NaN at ~1e37.

Eager LayerNorm stays stable; compile path emits NaN. Targets the
same kernel family as K33 with even larger inputs.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K34", title="LayerNorm compile vs eager near 1e37",
             issue=173793,
             expected="Compiled LayerNorm NaN at ~8.5e37 inputs; eager stable.")

import torch
import torch.nn as nn

x = torch.full((4, 256), 8.5e+37, dtype=torch.float32, device="cuda")
m = nn.LayerNorm(256).cuda().eval()

with torch.no_grad():
    y_eager = m(x)
m_c = torch.compile(m, dynamic=False)
with torch.no_grad():
    y_comp = m_c(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"eager  : nan={torch.isnan(y_eager).any().item()} inf={torch.isinf(y_eager).any().item()}")
print(f"compile: nan={torch.isnan(y_comp).any().item()} inf={torch.isinf(y_comp).any().item()}")
