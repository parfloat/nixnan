# Feedback for pytorch/pytorch#89757 — K12 — third-order gradient of `torch.pow` returns NaN on CUDA

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue89757/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue89757/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **INCONCLUSIVE**
- **Exception summary:** _log truncated past trailing report block — no aggregate available_
- **Top kernels launched:** `void at::native::elementwise_kernel<128, 2, at::native::gpu…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::elementwise_kernel<128, 2, at::native::gpu…`
- **Kernels emitting events:** `void at::native::reduce_kernel`, `void at::native::elementwise_kernel`, `void at::native::unrolled_elementwise_kernel`
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

**Killed at 3h22m CPU under SAMPLING=1** — `jacrev × 4` of `torch.pow` launches thousands of small autograd kernels and instrumenting each one didn't terminate in reasonable time. The 2.1 MB partial log captures the early per-event SASS but no trailing summary. A shallower autograd chain (third-order instead of fourth-order) might be a more tractable reproducer for future analysis.

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

- Issue README: [pytorch-issues/issue89757/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue89757/README.md)
- Reproducer: [pytorch-issues/issue89757/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue89757/data/repro.py)
- Captured trace: [pytorch-issues/issue89757/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue89757/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue89757/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue89757/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
