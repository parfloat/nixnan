"""K1 — pytorch/pytorch#181806: torch.signbit returns False for negative float16 NaN on CUDA.

Expected (per issue):
  cpu : tensor([True, True, True])
  cuda: tensor([False, False, False])
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K1', title='torch.signbit on float16 NaN', issue=181806, expected='CUDA torch.signbit should return True for negative-NaN fp16; it returns False.')

import numpy as np
import torch

src = torch.from_numpy(
    np.array([0xFE00, 0xFE00, 0xFE00], dtype=np.uint16).view(np.float16)
)

cpu = torch.signbit(src)
cuda = torch.signbit(src.cuda()).cpu()

print("torch:", torch.__version__)
print("cpu :", cpu)
print("cuda:", cuda)
print("divergence:", not torch.equal(cpu, cuda))
