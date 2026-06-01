"""K37 — pytorch/pytorch#181618: GroupNorm -> sum -> Conv1d under inductor heap-corrupts.

The crash is fault-on-free; nixnan won't catch glibc heap corruption
itself, but the surrounding kernels are interesting. We run the
compile path and let it either succeed, raise, or be killed.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K37", title="GroupNorm -> sum -> Conv1d under inductor",
             issue=181618,
             expected="Crash with 'free(): corrupted unsorted chunks' under torch.compile.")

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gn = nn.GroupNorm(4, 64)
        self.conv = nn.Conv1d(1, 8, 3, padding=1)

    def forward(self, x):
        x = self.gn(x)
        x = x.sum(dim=1, keepdim=True)
        return self.conv(x)

torch.manual_seed(0)
m = Net().cuda().eval()
x = torch.randn(2, 64, 16, device="cuda")
try:
    mc = torch.compile(m, dynamic=False, backend="inductor")
    with torch.no_grad():
        y = mc(x)
    torch.cuda.synchronize()
    print("torch:", torch.__version__)
    print("compiled output ok, shape:", y.shape)
except Exception as e:
    print("exception:", repr(e))
