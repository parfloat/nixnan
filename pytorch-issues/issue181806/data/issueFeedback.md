# Feedback for pytorch/pytorch#181806 — K1 — `torch.signbit` returns False for negative float16 NaN on CUDA

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k1_signbit_fp16_nan/` on the
`main` branch.

## What the trace observed

- **Verdict:** **SILENT-OUTSIDE-FP**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `void at::native::vectorized_elementwise_kernel`
- **Kernels emitting events:** _no exception events_
- **Final tensor / printed state: CPU/CUDA divergence**

## Recommendation (tentative)

The trace fires **zero NixNan arithmetic events** even though the bug reproduces (CPU returns `True`, CUDA returns `False` on a negative-NaN fp16). `torch.signbit` reads the sign bit without issuing any FP arithmetic, so the CUDA SASS falls outside NixNan's instrumentation scope. The reproduction signal is correct (the trace is silent), and the right verification target is a **bit-pattern equivalence check** between the CUDA and CPU kernels, not an exception-detection sweep.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k1_signbit_fp16_nan/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
