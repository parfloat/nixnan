"""K43 — pytorch/pytorch#175156: inductor randint inconsistent across runs.

Run randint twice via torch.compile to drive curand under instrumentation.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K43", title="inductor randint inconsistent RNG",
             issue=175156,
             expected="Multiple randint calls under torch.compile produce inconsistent RNG vs eager.")

import torch

def fn():
    a = torch.randint(0, 100, (1024,), device="cuda")
    b = torch.randint(0, 100, (1024,), device="cuda")
    return (a.float() * b.float()).sum()

torch.manual_seed(0)
r1 = fn()
torch.manual_seed(0)
r2 = fn()
torch.manual_seed(0)
fc = torch.compile(fn)
r3 = fc()
torch.cuda.synchronize()
print("torch:", torch.__version__)
print(f"eager r1={r1.item()}  r2={r2.item()}  identical={r1.item()==r2.item()}")
print(f"compile r3={r3.item()}  matches_eager={r1.item()==r3.item()}")
