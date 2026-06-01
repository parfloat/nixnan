"""K27 — pytorch/pytorch#173912: torch.distributions.Kumaraswamy mode is wrong.

The implementation computes (1-b)^(1/b) / (1-ab) instead of
((a-1)/(ab-1))^(1/a). Result is almost always NaN.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K27", title="Kumaraswamy distribution mode is wrong",
             issue=173912,
             expected="distribution.mode returns NaN where finite values are expected.")

import torch
from torch.distributions import Kumaraswamy

torch.set_default_dtype(torch.float64)

shapes = torch.tensor(
    [
        [0.5, 0.5], [1.0, 0.5], [0.5, 1.0], [1.0, 1.0],
        [3.0, 1.0], [1.0, 3.0], [3.0, 5.0],
        [1.000000001, 1.0000000000001], [2.0, 1e19],
    ], device="cuda",
)
a, b = shapes.T
distr = Kumaraswamy(a, b)
actual_mode = distr.mode
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("actual_mode:", actual_mode.cpu())
print("nan_count :", int(torch.isnan(actual_mode).sum().item()))
