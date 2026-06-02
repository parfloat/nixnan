# Feedback for pytorch/pytorch#182857 — K38 — SDPA backward `head_dim % 16 != 0`

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue182857/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182857/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **INCONCLUSIVE**
- **Exception summary:** _log truncated past trailing report block — no aggregate available_
- **Top kernels launched:** `ampere_sgemm_128x128_nn`, `void at::native::(anonymous namespace)::distribution_elemen…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

Heavy compile trace, 20 MB truncated past the summary block. We did not observe a crash on torch 2.3.1 (the bug was filed against torch.compile + SDPA backward, both of which have moved since). Same approach as K32: re-run with `PRINT_ILL_INSTR=0` for the summary, plus repeat on a fresher torch.

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

- Issue README: [pytorch-issues/issue182857/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182857/README.md)
- Reproducer: [pytorch-issues/issue182857/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182857/data/repro.py)
- Captured trace: [pytorch-issues/issue182857/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182857/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue182857/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue182857/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
