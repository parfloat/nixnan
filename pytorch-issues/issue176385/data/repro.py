"""K23 — pytorch/pytorch#176385: torch.erfinv float64 1D with both positive and negative out-of-domain.

PyTorch returns NaN for both signs; PaddlePaddle returns +inf / -inf.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K23", title="erfinv float64 out-of-domain",
             issue=176385,
             expected="CUDA erfinv on f64 returns NaN for |x|>=1; PaddlePaddle returns +/- inf.")

import torch

x = torch.tensor([-0.8771, -0.1721, 0.1404, 0.9607,  1.4295,
                  -0.4597, -1.4130,  1.8013, 1.0087, -0.0297],
                 dtype=torch.float64, device="cuda")

y = torch.erfinv(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("output:", y.cpu())
print("nan_count:", int(torch.isnan(y).sum().item()))
