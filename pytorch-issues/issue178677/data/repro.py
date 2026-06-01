"""K41 — pytorch/pytorch#178677: TransformerEncoder + all-True src_key_padding_mask.

Eager raises; torch.compile silently succeeds. We just run the eager
forward so the MultiheadAttention kernel runs under nixnan.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K41", title="TransformerEncoder all-masked src_key_padding_mask",
             issue=178677,
             expected="Eager raises RuntimeError; torch.compile silently succeeds (downstream NaN).")

import torch
import torch.nn as nn

torch.manual_seed(0)
enc_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True).cuda().eval()
enc = nn.TransformerEncoder(enc_layer, num_layers=1).cuda().eval()

B, N, D = 2, 16, 32
x = torch.randn(B, N, D, device="cuda")
mask = torch.ones(B, N, dtype=torch.bool, device="cuda")  # all positions masked

try:
    with torch.no_grad():
        y = enc(x, src_key_padding_mask=mask)
    torch.cuda.synchronize()
    print("torch:", torch.__version__)
    print(f"output: nan={torch.isnan(y).any().item()} inf={torch.isinf(y).any().item()}")
except Exception as e:
    print("exception:", repr(e))
