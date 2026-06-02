# Feedback for pytorch/pytorch#175154 — K44 — `F.interpolate(mode='nearest')` under inductor

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue175154/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue175154/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMLESS-STABLE**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `triton__0d1d2de`, `void at::native::reduce_kernel<512, 1, at::native::ReduceOp…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

**Zero events.** The bug is a correctness issue (wrong upsampled values), not an IEEE FP exception. NixNan won't surface wrong-but-finite results; only a reference-comparison test would. Outside this tool's scope.

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

- Issue README: [pytorch-issues/issue175154/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue175154/README.md)
- Reproducer: [pytorch-issues/issue175154/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue175154/data/repro.py)
- Captured trace: [pytorch-issues/issue175154/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue175154/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue175154/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue175154/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
