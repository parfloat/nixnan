# Feedback for pytorch/pytorch#181618 — K37 — `GroupNorm -> sum -> Conv1d` under inductor

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue181618/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181618/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMFUL-LOCAL**
- **Exception summary:** **fp32** sub=3
- **Top kernels launched:** `triton__0d1d2de`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::`
- **Kernels emitting events:** `triton__0d1d2d3e4de at /home/ganesh/.local/lib/python3.10/s…`
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

The trace shows **3 fp32 subnormal events** on the GroupNorm + sum + Conv1d chain — and notably, **no heap corruption** on our torch 2.3.1 build. The subnormal events suggest the underflow path inside GroupNorm's variance reduction is exercised even on benign inputs, which is consistent with the crash being downstream of an underflow corner case. A more extensive subnormal-aware test on the GroupNorm fusion path might surface the heap-corruption trigger.

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

- Issue README: [pytorch-issues/issue181618/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181618/README.md)
- Reproducer: [pytorch-issues/issue181618/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181618/data/repro.py)
- Captured trace: [pytorch-issues/issue181618/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181618/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue181618/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181618/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
