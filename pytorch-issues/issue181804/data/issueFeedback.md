# Feedback for pytorch/pytorch#181804 — K2 — `torch.copysign` ignores sign bit of negative float16 NaN on CUDA

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue181804/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181804/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **SILENT-OUTSIDE-FP**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `void at::native::vectorized_elementwise_kernel`
- **Kernels emitting events:** _no exception events_
- **Final tensor / printed state: CPU/CUDA divergence**

## Recommendation (tentative)

Same architectural boundary as #181806 — the trace fires zero NixNan events because `copysign` manipulates the sign bit without issuing FP arithmetic. The bug reproduces (`cpu: [1,-1,1]`, `gpu: [1,1,1]`) but is invisible to FP instrumentation. A bit-pattern check against CPU on the input's sign-bit operand inside the CUDA kernel is the right test.

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

- Issue README: [pytorch-issues/issue181804/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181804/README.md)
- Reproducer: [pytorch-issues/issue181804/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181804/data/repro.py)
- Captured trace: [pytorch-issues/issue181804/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181804/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue181804/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181804/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
