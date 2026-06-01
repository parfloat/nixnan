# Feedback for pytorch/pytorch#176384 — K9 — `torch.erfinv` on out-of-domain input returns NaN on CUDA

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k9_erfinv_out_of_domain/` on the
`main` branch.

## What the trace observed

- **Verdict:** **HARMFUL-PROPAGATED**
- **Exception summary:** **fp32** NaN=29
- **Top kernels launched:** `erfinv_kernel_vectorized4_kernel`
- **Kernels emitting events:** `erfinv_kernel_vectorized4_kernel`
- **Final tensor / printed state: NaN**

## Recommendation (tentative)

The trace shows **29 fp32 NaN events** on the erfinv kernel — expected since `erfinv` is mathematically undefined for `|x| >= 1`. The choice of NaN vs ±Inf for out-of-domain inputs is a framework convention call; PyTorch's NaN is defensible (NaN signals 'undefined'), and the upstream comparison with PaddlePaddle's `-inf` is more about cross-framework consistency than IEEE correctness.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k9_erfinv_out_of_domain/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
