"""K22 — pytorch/pytorch#176412: torch.erfinv on 2D float32 with negative out-of-domain.

erfinv's domain is (-1, 1). For input values like -1.4968 PyTorch returns
NaN; PaddlePaddle returns -inf. We just exercise the CUDA erfinv kernel.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K22", title="erfinv 2D float32 negative out-of-domain",
             issue=176412,
             expected="CUDA erfinv emits NaN for out-of-domain inputs; PaddlePaddle returns -inf.")

import torch

x = torch.tensor([[0.2597], [-0.9840], [-0.2840],
                  [0.9317], [-1.0186], [-1.6230],
                  [-0.2458], [-1.4968], [-0.9785],
                  [0.8734]], dtype=torch.float32, device="cuda")

y = torch.erfinv(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("output:\n", y.cpu())
print("nan_count:", int(torch.isnan(y).sum().item()))
