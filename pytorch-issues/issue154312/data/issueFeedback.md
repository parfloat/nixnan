# Feedback for pytorch/pytorch#154312 — K7 — `torch.logdet` incorrect for singular matrix on CUDA

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue154312/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue154312/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMLESS-STABLE**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `xxtrf4_set_info_ker`, `void getrf_pivot`, `void ipiv_lower_small`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

Our test matrix had `det = 2.68e-7` — non-zero in fp32, so the cuSOLVER path computed a finite (wrong) `-15.13` cleanly and **zero** NixNan events fired. The bug is real (the matrix is rank-deficient in exact arithmetic; CPU correctly returns `-inf`), but the fp32 representation of the determinant narrowly avoids the singular threshold cuSOLVER uses. A more robust singularity check in the CUDA logdet path — or a warning when the determinant magnitude is within several ULPs of zero — would surface this class of mismatch. Worth rechecking with the exact-rank-deficient inputs from your upstream report.

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

- Issue README: [pytorch-issues/issue154312/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue154312/README.md)
- Reproducer: [pytorch-issues/issue154312/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue154312/data/repro.py)
- Captured trace: [pytorch-issues/issue154312/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue154312/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue154312/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue154312/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
