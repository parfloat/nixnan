"""K2 — pytorch/pytorch#181804: torch.copysign ignores sign bit of negative float16 NaN on CUDA.

Expected:
  cpu: tensor([ 1., -1.,  1.], dtype=torch.float16)
  gpu: tensor([1., 1., 1.], dtype=torch.float16)
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K2', title='torch.copysign on float16 NaN sign tensor', issue=181804, expected='CUDA ignores the sign bit of the negative-NaN second operand.')

import numpy as np
import torch

mag = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16)
sgn = torch.from_numpy(
    np.array([0x7E00, 0xFE00, 0x3C00], dtype=np.uint16).view(np.float16)
)

cpu = torch.copysign(mag, sgn)
gpu = torch.copysign(mag.cuda(), sgn.cuda()).cpu()

print("torch:", torch.__version__)
print("cpu:", cpu)
print("gpu:", gpu)
print("divergence:", not torch.equal(cpu, gpu))
