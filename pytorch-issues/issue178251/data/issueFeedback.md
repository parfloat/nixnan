# Feedback for pytorch/pytorch#178251 — K25 — SDPA backward NaN with attn_mask

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue178251/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178251/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMFUL-LOCAL**
- **Exception summary:** **fp16** ±Inf=128; **fp32** ±Inf=4796
- **Top kernels launched:** `void at::native::reduce_kernel<512, 1, at::native::ReduceOp…`, `void at::native::elementwise_kernel<128, 2, at::native::gpu…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

The trace shows **4,796 fp32 ±Inf events plus 128 fp16 ±Inf events** from the SDPA backward kernel under `EFFICIENT_ATTENTION` with a partial-`-inf` `attn_mask`. The ±Inf events are consistent with the score gradient through fully-masked positions — `softmax_grad` × `(-inf_mask)` produces `0 × -inf = NaN` then downstream FFMAs propagate. A NaN-safe gradient (zero gradient through fully-masked rows) would eliminate the entire 4,924-event cascade.

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

- Issue README: [pytorch-issues/issue178251/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178251/README.md)
- Reproducer: [pytorch-issues/issue178251/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178251/data/repro.py)
- Captured trace: [pytorch-issues/issue178251/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178251/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue178251/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178251/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
