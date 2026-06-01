"""K36 — pytorch/pytorch#181692: nn.RReLU().eval() wrong under torch.compile.

In eval mode RReLU should behave deterministically (uses mean of bounds)
but compiled output differs from eager. Tests Inductor's RReLU lowering.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K36", title="RReLU eval under inductor",
             issue=181692,
             expected="Compiled RReLU().eval() produces a different result than eager.")

import torch
import torch.nn as nn

torch.manual_seed(0)
m = nn.RReLU().eval().cuda()
x = torch.randn(1024, device="cuda")
y_eager = m(x)
mc = torch.compile(m, dynamic=False)
y_comp = mc(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"eager  sum: {y_eager.sum().item()}")
print(f"compile sum: {y_comp.sum().item()}")
print(f"max abs diff: {(y_eager - y_comp).abs().max().item()}")
