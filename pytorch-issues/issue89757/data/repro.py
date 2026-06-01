"""K12 — pytorch/pytorch#89757: third-order gradient of torch.pow returns NaN.

torch.pow with tensor power and zero base produces NaN at higher-order autograd.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid='K12', title='torch.pow third-order gradient', issue=89757, expected='Higher-order grad of pow at zero returns all NaN instead of expected finite values.')

import torch
from torch.func import jacrev

def bar(x, y):
    power = torch.tensor([1, 1, 1, 1], dtype=torch.int64, device=x.device)
    state = torch.cat([x, y])
    state = torch.pow(state, power)
    return torch.stack([state.prod(-1)])

dev = "cuda"
x = torch.tensor([0.0, 0.0], dtype=torch.float64, device=dev)
y = torch.tensor([0.0, 0.0], dtype=torch.float64, device=dev)

out = bar(x, y)
g4 = jacrev(jacrev(lambda x: jacrev(jacrev(lambda y: bar(x, y)))(y)))(x).flatten().cpu()
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("forward:", out.cpu())
print("4th-order grad:", g4)
print("any_nan:", torch.isnan(g4).any().item())
