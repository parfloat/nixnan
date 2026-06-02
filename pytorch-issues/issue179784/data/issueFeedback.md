# Feedback for pytorch/pytorch#179784 — K10 — `torch.xlogy(0, 0)` returns NaN on CUDA

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue179784/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue179784/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMLESS-STABLE**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `void at::native::vectorized_elementwise_kernel`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

We observed **zero NixNan events**: on torch 2.3.1 `xlogy(0, 0)` returns `0` cleanly, not `NaN`. Either the implementation was rewritten to special-case the zero argument, or our 2.3.1 build differs from the version the bug was filed against. Confirming with `git log aten/src/ATen/native/cuda/XlogyKernel.cu` against your torch version would clarify whether the fix is already in place upstream.

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

- Issue README: [pytorch-issues/issue179784/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue179784/README.md)
- Reproducer: [pytorch-issues/issue179784/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue179784/data/repro.py)
- Captured trace: [pytorch-issues/issue179784/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue179784/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue179784/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue179784/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
