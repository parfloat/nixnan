"""K35 — pytorch/pytorch#182131: Inductor produces different result for fp16 cast + elementwise add.

Compiles a small fn that casts to fp16 then adds; result differs from
eager. Exercises Inductor's lowering of the cast op.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K35", title="torch.compile fp16 cast + add",
             issue=182131,
             expected="Inductor lowering of fp16 cast before elementwise add gives a different result than eager.")

import torch

torch.manual_seed(0)
def f(a, b):
    return a.to(torch.float16) + b.to(torch.float16)

a = torch.randn(8192, device="cuda")
b = torch.randn(8192, device="cuda")
y_eager = f(a, b)
fc = torch.compile(f)
y_comp = fc(a, b)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"eager  : nan={torch.isnan(y_eager).any().item()} max={y_eager.max().item()}")
print(f"compile: nan={torch.isnan(y_comp).any().item()}  max={y_comp.max().item()}")
print(f"diverge: {(y_eager != y_comp).any().item()}")
