"""K24 — pytorch/pytorch#173520: Conv2d overflow behaviour diverges CUDA vs CPU.

Upstream uses an attached repro.zip with specific input.pt / weight_bias.pt;
we substitute synthetic large-magnitude input/weights that drive the
accumulator near FLT_MAX so the cuDNN convolution still has work to do.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K24", title="Conv2d overflow CUDA vs CPU",
             issue=173520,
             expected="CUDA cuDNN convolution overflow behaviour differs from CPU near FLT_MAX.")

import torch
import torch.nn as nn

torch.manual_seed(0)
# 3.0e+38 * 3.0 ~ FLT_MAX boundary
inp = torch.randn(1, 3, 32, 32) * 3.0e+37

conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=True)
conv.weight.data.fill_(0.5)
conv.bias.data.zero_()
conv.eval()

inp_cuda = inp.cuda()
conv.cuda()
with torch.no_grad():
    out_cuda = conv(inp_cuda).cpu()
conv.cpu()
with torch.no_grad():
    out_cpu = conv(inp.cpu())

torch.cuda.synchronize()
print("torch:", torch.__version__)
print(f"CUDA  out range: [{out_cuda.min().item():.2e}, {out_cuda.max().item():.2e}]  nan={torch.isnan(out_cuda).any().item()}  inf={torch.isinf(out_cuda).any().item()}")
print(f"CPU   out range: [{out_cpu.min().item():.2e}, {out_cpu.max().item():.2e}]   nan={torch.isnan(out_cpu).any().item()}   inf={torch.isinf(out_cpu).any().item()}")
