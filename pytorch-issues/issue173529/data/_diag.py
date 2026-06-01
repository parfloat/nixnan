"""Shared diagnostic helper for the K1-K20 reproducers.

Goal: every stdout.nnlog ends up self-describing — kernel id, upstream
issue link, expected behaviour, input/output tensor stats, and a final
divergence verdict — so the committed nnlogs are root-cause-ready
without needing to cross-reference the repo.

Usage in each repro:

    import _diag
    _diag.banner(kid="K3", title="...", issue=178000, expected="...")
    ...
    _diag.stats("input  A", A)
    _diag.stats("output gpu", gpu_result)
    _diag.diverged("logabsdet", cpu_result, gpu_result)
"""
from __future__ import annotations

import platform
import sys


def banner(kid: str, title: str, issue: int | str | None, expected: str) -> None:
    import torch
    print(f"=== {kid}: {title}")
    if issue is not None:
        print(f"=== issue : https://github.com/pytorch/pytorch/issues/{issue}")
    print(f"=== expect: {expected}")
    print(f"=== torch : {torch.__version__}    python: {platform.python_version()}")
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        print(f"=== device: {torch.cuda.get_device_name(0)}  sm_{cc[0]}{cc[1]}")
    else:
        print("=== device: cuda NOT available — repro will fail")
    sys.stdout.flush()


def stats(name: str, t) -> None:
    """One-line summary of a tensor: shape, dtype, min, max, nan/inf counts."""
    import torch
    try:
        if t.dtype.is_complex:
            real = t.real
            min_v = real.min().item()
            max_v = real.max().item()
        elif t.dtype.is_floating_point:
            min_v = t.detach().min().item()
            max_v = t.detach().max().item()
        else:
            min_v = t.detach().min().item()
            max_v = t.detach().max().item()
        nan_n = int(torch.isnan(t).sum().item()) if t.dtype.is_floating_point or t.dtype.is_complex else 0
        inf_n = int(torch.isinf(t).sum().item()) if t.dtype.is_floating_point or t.dtype.is_complex else 0
        msg = (f"shape={list(t.shape)} dtype={t.dtype} dev={t.device} "
               f"min={min_v:.6g} max={max_v:.6g} nan={nan_n} inf={inf_n}")
    except Exception as e:
        msg = f"<stats failed: {type(e).__name__}: {e}>"
    print(f"  {name}: {msg}")
    sys.stdout.flush()


def diverged(label: str, cpu_t, gpu_t) -> bool:
    """Compare a CPU result against the same op's CUDA result; emit one
    line summarising whether the upstream bug's CPU-vs-CUDA divergence
    is reproduced on this run."""
    import torch
    cpu = cpu_t.detach().to("cpu")
    gpu = gpu_t.detach().to("cpu")
    if cpu.shape != gpu.shape:
        print(f"=== diverged[{label}]: SHAPE MISMATCH cpu={list(cpu.shape)} gpu={list(gpu.shape)}")
        return True
    if cpu.dtype.is_floating_point and gpu.dtype.is_floating_point:
        nan_mismatch = (torch.isnan(cpu) != torch.isnan(gpu)).any().item()
        inf_mismatch = (torch.isinf(cpu) != torch.isinf(gpu)).any().item()
        finite_mask = torch.isfinite(cpu) & torch.isfinite(gpu)
        if finite_mask.any():
            cpu_f = cpu[finite_mask].double()
            gpu_f = gpu[finite_mask].double()
            max_abs_diff = (cpu_f - gpu_f).abs().max().item()
        else:
            max_abs_diff = float("nan")
        diverged_flag = nan_mismatch or inf_mismatch or max_abs_diff > 1e-4
        print(f"=== diverged[{label}]: {diverged_flag}  "
              f"nan_mask_mismatch={nan_mismatch} inf_mask_mismatch={inf_mismatch} "
              f"max_abs_diff_finite={max_abs_diff:.6g}")
    else:
        same = torch.equal(cpu, gpu)
        print(f"=== diverged[{label}]: {not same}")
        diverged_flag = not same
    sys.stdout.flush()
    return diverged_flag
