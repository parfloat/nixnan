"""K21 — pytorch/pytorch#176415: batched torch.logdet on 4D float32.

Cross-framework migration showed PyTorch returning NaN only for the
non-positive-determinant matrices while TensorFlow returned NaN across
the whole batch. Here we just exercise the batched cuSOLVER logdet
path on CUDA so nixnan can instrument the LU decompositions.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K21", title="batched logdet 4D float32",
             issue=176415,
             expected="CUDA cuSOLVER batched logdet emits NaN only at non-positive-determinant batches.")

import numpy as np
import torch

np.random.seed(42)
A_np = np.random.randn(4, 4, 2, 2).astype(np.float32)
A = torch.from_numpy(A_np).cuda()

out = torch.logdet(A)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("input shape:", A.shape, A.dtype)
print("output:", out.cpu())
print("nan_count:", int(torch.isnan(out).sum().item()))
print("inf_count:", int(torch.isinf(out).sum().item()))
