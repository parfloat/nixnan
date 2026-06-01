"""K31 — pytorch/pytorch#182663: torch.arange divide-by-zero FPE.

Upstream filed against CPU (compute_arange_size division). On CUDA the
arange kernel needs a valid size pre-computed on the host; passing
step=0 to torch.arange may raise (or be silently sanitised). Either
way we run arange on CUDA to drive that kernel.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K31", title="torch.arange divide-by-zero",
             issue=182663,
             expected="torch.arange with step=0 should raise; CPU FPE crash on some builds.")

import torch

print("torch:", torch.__version__)
# Step != 0 baseline — runs the arange CUDA kernel
x = torch.arange(0, 1024, 1, device="cuda", dtype=torch.float32)
print("baseline arange shape:", x.shape, "sum:", x.sum().item())

# Step = 0 — this triggers the upstream bug
try:
    y = torch.arange(0, 10, 0, dtype=torch.float32, device="cuda")
    print("step=0 surprisingly produced:", y)
except Exception as e:
    print("step=0 raised:", repr(e))

torch.cuda.synchronize()
