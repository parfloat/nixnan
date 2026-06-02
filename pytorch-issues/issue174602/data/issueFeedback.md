# Feedback for pytorch/pytorch#174602 — K28 — `ctc_loss` inconsistency

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue174602/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue174602/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMFUL-LOCAL**
- **Exception summary:** **fp64** NaN=1962 ±Inf=630 sub=18
- **Top kernels launched:** `void at::native::(anonymous namespace)::ctc_loss_log_alpha_…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::`
- **Kernels emitting events:** `void at::native::`
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

The trace shows **1,962 fp64 NaN, 630 ±Inf, and 18 subnormal events** from the cuDNN CTC kernel. The high subnormal count is unusual in this corpus — only K4 (91), K33 (6), and K47 (0) approach it — and is consistent with an underflow chain in the forward-backward `alpha` / `beta` recursion. The non-determinism upstream describes might be partially downstream of these underflows: subnormal values are particularly sensitive to FMA reordering by the scheduler. Worth comparing the trace against the deterministic-algorithm code path to see if the subnormal count differs.

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

- Issue README: [pytorch-issues/issue174602/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue174602/README.md)
- Reproducer: [pytorch-issues/issue174602/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue174602/data/repro.py)
- Captured trace: [pytorch-issues/issue174602/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue174602/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue174602/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue174602/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
