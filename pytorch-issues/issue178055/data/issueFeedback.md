# Feedback for pytorch/pytorch#178055 — K32 — `torch.compile` Inf on fp32 boundary

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue178055/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178055/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **INCONCLUSIVE**
- **Exception summary:** _log truncated past trailing report block — no aggregate available_
- **Top kernels launched:** `void implicit_convolve_sgemm`
- **Kernels emitting events:** _no exception events_
- **Final tensor / printed state: Inf**

## Recommendation (tentative)

Trace truncated past the 20 MB natural ceiling — Inductor's Triton trace volume for this Conv2d is heavy. To capture the summary block we would need a second pass with `PRINT_ILL_INSTR=0` or a coarser bin spec. The K33 / K34 siblings (LayerNorm under torch.compile) reproduce cleanly on torch 2.3.1, which suggests Inductor's normalisation lowering shares the same fp32-boundary blind spot.

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

- Issue README: [pytorch-issues/issue178055/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178055/README.md)
- Reproducer: [pytorch-issues/issue178055/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178055/data/repro.py)
- Captured trace: [pytorch-issues/issue178055/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178055/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue178055/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178055/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
