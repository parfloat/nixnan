"""K29 — pytorch/pytorch#181684: DISABLED test_addcdiv_cuda_float16.

Test is flaky in CI; we lift a minimal float16 addcdiv call onto CUDA
so nixnan can see the underlying SASS — addcdiv is value = input +
scalar * tensor1 / tensor2 which can underflow / produce NaN in fp16.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K29", title="addcdiv fp16 on CUDA",
             issue=181684,
             expected="addcdiv on fp16 stresses divide + scaled-add SASS; may emit NaN/Inf.")

import torch

torch.manual_seed(0)
a = torch.randn(1024, dtype=torch.float16, device="cuda")
b = torch.randn(1024, dtype=torch.float16, device="cuda")
c = torch.randn(1024, dtype=torch.float16, device="cuda")
# inject zero divisor in c to drive divide-by-zero
c[100] = 0.0
c[200] = 0.0

y = torch.addcdiv(a, b, c, value=1.5)
torch.cuda.synchronize()
print("torch:", torch.__version__)
print(f"y stats: nan={torch.isnan(y).any().item()} inf={torch.isinf(y).any().item()}")
print(f"y range [{y.min().item()}, {y.max().item()}]")
