# Feedback for pytorch/pytorch#173799 — K5 — `F.pdist(p=0)` disagrees CPU vs CUDA on inf input

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k5_pdist_p0_inf/` on the
`main` branch.

## What the trace observed

- **Verdict:** **HARMFUL-PROPAGATED**
- **Exception summary:** **fp64** NaN=80 ±Inf=200
- **Top kernels launched:** `void at::native::`
- **Kernels emitting events:** `void at::native::`
- **Final tensor / printed state: CPU/CUDA divergence**

## Recommendation (tentative)

The trace shows **80 fp64 NaN and 200 ±Inf events** matching the `inf - inf` pairwise difference described in the issue. The CPU implementation propagates the NaN; the CUDA kernel treats the NaN difference as a non-zero L0 count. Adding the same NaN-aware path to the CUDA pdist L0 reduction would close the gap. The 200 ±Inf events confirm the intermediate inf is computed (then implicitly squashed by the L0 count).

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k5_pdist_p0_inf/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
