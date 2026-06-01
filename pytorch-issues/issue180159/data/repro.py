"""K30 — pytorch/pytorch#180159: float NaN/Inf cast to int — CPU vs GPU disagree.

CPU produces INT_MIN; GPU produces 0 or INT_MAX. Tests the cast kernel.
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _diag  # noqa: E402
_diag.banner(kid="K30", title="float NaN/Inf cast to int",
             issue=180159,
             expected="CUDA cast of NaN/Inf to int returns saturated values while CPU returns INT_MIN.")

import torch

cases = [
    (float("nan"), torch.int32, "float32(nan) -> int32"),
    (float("inf"), torch.int32, "float32(inf) -> int32"),
    (float("inf"), torch.int64, "float32(inf) -> int64"),
]
print("torch:", torch.__version__)
for val, dtype, name in cases:
    x = torch.tensor(val, dtype=torch.float32)
    cpu_v = x.to(dtype).item()
    gpu_v = x.cuda().to(dtype).cpu().item()
    print(f"  {name}: CPU={cpu_v}  GPU={gpu_v}  divergent={cpu_v != gpu_v}")

torch.cuda.synchronize()
