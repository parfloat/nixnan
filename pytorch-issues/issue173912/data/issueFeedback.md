# Feedback for pytorch/pytorch#173912 — K27 — `Kumaraswamy.mode` returns NaN

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue173912/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173912/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **INCONCLUSIVE**
- **Exception summary:** _log truncated past trailing report block — no aggregate available_
- **Top kernels launched:** `void at::native::elementwise_kernel<128, 4, at::native::gpu…`, `void at::native::elementwise_kernel<128, 2, at::native::gpu…`, `void at::native::reduce_kernel<512, 1, at::native::ReduceOp…`
- **Kernels emitting events:** `void at::native::vectorized_elementwise_kernel`, `void at::native::elementwise_kernel`, `void at::native::unrolled_elementwise_kernel`
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

The reproducer was **killed at ~1.5h CPU** under SAMPLING=1; the partial 3.2 MB log has no trailing summary. Notably, even a complete trace likely wouldn't flag the bug — the issue is a wrong formula (`(1-b)^(1/b) / (1-ab)` vs the correct `((a-1)/(ab-1))^(1/a)`), which is a math-correctness issue rather than an IEEE-exception event. NixNan would only fire on the boundary cases (when one of the divisors is 0 or the input is itself non-finite). The math fix described in the issue body is the right path independent of nixnan.

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

- Issue README: [pytorch-issues/issue173912/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173912/README.md)
- Reproducer: [pytorch-issues/issue173912/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173912/data/repro.py)
- Captured trace: [pytorch-issues/issue173912/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173912/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue173912/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173912/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
