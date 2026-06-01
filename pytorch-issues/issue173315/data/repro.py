"""K26 — pytorch/pytorch#173315: nn.LSTM produces NaN on CUDA eager.

Upstream needed bundle.pt + lstm1_weights.pt attachments. We substitute
random init plus very-large inputs so the cuDNN RNN kernel still does
substantial work and may produce non-finite outputs.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K26", title="nn.LSTM CUDA eager (random init)",
             issue=173315,
             expected="CUDA cuDNN LSTM may emit NaN for very-large inputs even when CPU stays finite.")

import torch
import torch.nn as nn

torch.manual_seed(0)
H = 50
T = 200
B = 4
lstm = nn.LSTM(input_size=H, hidden_size=H, num_layers=1, batch_first=True).cuda().eval()
x = torch.randn(B, T, H, device="cuda") * 1e10

with torch.no_grad():
    y, _ = lstm(x)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print(f"y stats: shape={list(y.shape)} nan={torch.isnan(y).any().item()} inf={torch.isinf(y).any().item()}")
print(f"y range: [{y.min().item():.3e}, {y.max().item():.3e}]")
