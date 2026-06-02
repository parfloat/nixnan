# Feedback for pytorch/pytorch#180159 — K30 — float NaN/Inf cast to int

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue180159/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue180159/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMLESS-STABLE**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `void at::native::unrolled_elementwise_kernel`, `void at::native::unrolled_elementwise_kernel<at::native::di…`, `void at::native::unrolled_elementwise_kernel<at::native::di…`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

The trace fires **zero events** even though the bug reproduces (CPU `nan -> INT_MIN`, CUDA `nan -> 0`). Type cast from float to int doesn't issue FP arithmetic on the integer side, so NixNan's instrumentation doesn't see it. This is the third repro in the corpus (alongside K1 / K2 signbit / copysign) where NixNan's silence is structural rather than diagnostic. A `nan_to_num` step before the cast, or a documented saturation policy for the CUDA cast kernel, would eliminate the CPU/CUDA disagreement.

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

- Issue README: [pytorch-issues/issue180159/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue180159/README.md)
- Reproducer: [pytorch-issues/issue180159/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue180159/data/repro.py)
- Captured trace: [pytorch-issues/issue180159/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue180159/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue180159/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue180159/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
