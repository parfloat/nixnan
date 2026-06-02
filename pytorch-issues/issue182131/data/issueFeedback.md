# Feedback for pytorch/pytorch#182131 — K35 — `torch.compile` fp16 cast + add

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue182131/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182131/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMLESS-STABLE**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `triton__0d1d2d3de`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

**Zero events on torch 2.3.1.** The Inductor lowering of `tensor.to(fp16) + tensor.to(fp16)` agrees with eager on the older toolchain. The bug appears to be specific to torch ≥ 2.10; a sub-sweep on a fresher PyTorch wheel (cu126 or cu128) would reactivate it.

## A few modest questions back to you

1. Does this trace plus the analysis above help narrow the root cause
   for you now, or do you already have a confirmed cause that this
   confirms / contradicts?
2. Would such a trace have helped at the time you originally filed
   this issue — i.e. would it have shortened your "what is actually
   wrong inside CUDA?" search?
3. Was this issue a show-stopper for your work, or filed primarily
   as a best-practice / correctness flag for the broader PyTorch
   community?

## Where to find the artifacts

- Issue README: [pytorch-issues/issue182131/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182131/README.md)
- Reproducer: [pytorch-issues/issue182131/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182131/data/repro.py)
- Captured trace: [pytorch-issues/issue182131/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182131/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue182131/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182131/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
