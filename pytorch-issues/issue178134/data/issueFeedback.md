# Feedback for pytorch/pytorch#178134 — K45 — `torch.compile` Conv2d fp32 drift

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue178134/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178134/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **INCONCLUSIVE**
- **Exception summary:** _log truncated past trailing report block — no aggregate available_
- **Top kernels launched:** `void at::native::`, `void at::native::vectorized_elementwise_kernel`, `void implicit_convolve_sgemm`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

Trace truncated past the 20 MB summary. The Inductor Conv2d lowering ran heavily but the summary block was lost. Same approach as K32 / K38: re-run with `PRINT_ILL_INSTR=0` to capture the aggregate counts. The drift behaviour described upstream is at smaller magnitudes than K32's FLT_MAX, but our trace can't yet tell whether the events would fall in the same kernel.

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

- Issue README: [pytorch-issues/issue178134/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178134/README.md)
- Reproducer: [pytorch-issues/issue178134/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178134/data/repro.py)
- Captured trace: [pytorch-issues/issue178134/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178134/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue178134/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178134/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
