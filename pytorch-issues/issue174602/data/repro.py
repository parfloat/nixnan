"""K28 — pytorch/pytorch#174602: ctc_loss returns inconsistent results.

Upstream reproduces non-determinism by repeating the call across
multiple compile cache invocations. We just run the eager CUDA path
twice and let nixnan instrument the underlying CTC kernel.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K28", title="ctc_loss inconsistency",
             issue=174602,
             expected="ctc_loss output sometimes inconsistent across runs; cuDNN CTC kernel.")

import torch

torch.manual_seed(2026)
log_probs = torch.randn(50, 3, 15, dtype=torch.float64, device="cuda")
targets = torch.randint(1, 100, (3, 30), dtype=torch.int64, device="cuda")
input_lengths = [50, 50, 50]
target_lengths = [30, 25, 20]

# Run twice eagerly; both should be identical
r1 = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
r2 = torch.nn.functional.ctc_loss(log_probs.clone(), targets, input_lengths, target_lengths)
torch.cuda.synchronize()

print("torch:", torch.__version__)
print("r1:", r1.item())
print("r2:", r2.item())
print("identical:", torch.equal(r1, r2))
