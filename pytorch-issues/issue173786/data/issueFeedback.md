# Feedback for pytorch/pytorch#173786 — K4 — `torch.linalg.cholesky_ex`: silent NaN on CUDA for `inf` input

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k4_cholesky_ex_inf/` on the
`main` branch.

## What the trace observed

- **Verdict:** **HARMFUL-LOCAL**
- **Exception summary:** **fp64** NaN=19942 ±Inf=1040 sub=91 ÷0=819
- **Top kernels launched:** `xxtrf4_set_info_ker`, `void kernel`, `void at::native::triu_tril_kernel`
- **Kernels emitting events:** `void kernel`
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

The trace shows **19,942 fp64 NaN, 988 ±Inf, 91 subnormal, and 819 explicit divide-by-zero events** from cuSOLVER's Cholesky path on a 1×1 `[[inf]]` matrix. The 819 ÷0 events are the canonical signature of the bidiagonalisation step dividing by an inf-derived pivot; the 91 subnormal events are downstream products of `inv(inf) = 0`. Two minimal candidate fixes: (a) an input-validation short-circuit (if any input is non-finite, return inf with `info=0`), or (b) NaN-safe behaviour on the reciprocal pivot step. Either would prevent the silent NaN with `info=0` that the upstream report describes.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k4_cholesky_ex_inf/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
