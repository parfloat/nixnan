"""K42 — pytorch/pytorch#177821: inductor drops a complex indexing assignment.

The compiled version skips the indexing write. We just run both and
diff so the Inductor-emitted kernel runs.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K42", title="inductor complex indexing assignment",
             issue=177821,
             expected="Inductor compiled version silently drops an indexing assignment.")

import torch

torch.manual_seed(0)

def fn(x):
    y = x.clone()
    idx = (y < 0)
    y[idx] = -y[idx] * 2.0
    return y

x = torch.randn(1024, device="cuda")
y_eager = fn(x)
fc = torch.compile(fn)
y_comp = fc(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"max abs diff: {(y_eager - y_comp).abs().max().item()}")
print(f"max eager: {y_eager.max().item()} max compile: {y_comp.max().item()}")
