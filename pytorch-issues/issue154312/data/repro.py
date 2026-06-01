"""K7 — pytorch/pytorch#154312: torch.logdet incorrect on singular matrix on CUDA.

CPU returns -inf (correct, since det = 0); CUDA returns a finite value.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K7', title='torch.logdet on singular matrix', issue=154312, expected='CUDA returns finite logdet for a singular matrix; CPU returns -inf.')

import torch

# Row 3 = 3 * Row 1 -> singular
A = torch.tensor([[1.0, 2.0, 3.0],
                  [2.0, 5.0, 6.0],
                  [3.0, 6.0, 9.0]], dtype=torch.float32)

cpu_result = A.logdet()
gpu_result = A.cuda().logdet()
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"CPU  logdet : {cpu_result.item()}")
print(f"CUDA logdet : {gpu_result.item()}")
print(f"det         : {A.det().item()}")
print("divergence:", cpu_result.item() != gpu_result.item())
