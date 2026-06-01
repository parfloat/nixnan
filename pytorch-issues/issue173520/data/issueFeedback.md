# Feedback for pytorch/pytorch#173520 — K24 — `nn.Conv2d` near `FLT_MAX` overflow

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k24_conv2d_overflow_divergence/` on the
`main` branch.

## What the trace observed

- **Verdict:** **HARMLESS-STABLE**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `void implicit_convolve_sgemm`, `void at::native::elementwise_kernel`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

Our synthetic `3.0e+37 * 0.5` Conv2d input did not actually cross FLT_MAX once accumulated, so the trace shows **zero events**. The upstream `repro.zip` attachment is the authoritative reproducer; without it the bug doesn't reliably manifest. Embedding the exact input tensor values in the issue body (perhaps as a base64-encoded `.pt`) would let third parties reproduce without the attachment.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k24_conv2d_overflow_divergence/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
