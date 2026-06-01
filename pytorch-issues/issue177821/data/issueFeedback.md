# Feedback for pytorch/pytorch#177821 — K42 — inductor complex indexing assignment

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k42_inductor_complex_indexing/` on the
`main` branch.

## What the trace observed

- **Verdict:** **HARMLESS-STABLE**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `triton__0d1d2d3de`, `triton__0d1d2`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

**Zero events on torch 2.3.1.** The Inductor lowering matches eager on this PyTorch version. Bug appears to be specific to torch ≥ 2.10; this trace doesn't add diagnostic value until re-run on a fresher wheel.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k42_inductor_complex_indexing/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
