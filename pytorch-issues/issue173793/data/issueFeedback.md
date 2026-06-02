# Feedback for pytorch/pytorch#173793 — K34 — LayerNorm `torch.compile` near `1e37`

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue173793/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173793/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMFUL-PROPAGATED**
- **Exception summary:** **fp32** NaN=1676 ±Inf=132; mem-NaN fp32=2
- **Top kernels launched:** `void at::native::reduce_kernel<512, 1, at::native::ReduceOp…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`
- **Kernels emitting events:** `void at::native::`, `void at::native::vectorized_elementwise_kernel`, `void at::native::unrolled_elementwise_kernel`
- **Final tensor / printed state: NaN**

## Recommendation (tentative)

Sibling of #178084 (our K33) with even larger inputs (~`1e37`). The trace shows **1,676 fp32 NaN, 132 ±Inf, and 2 mem-NaN events** — slightly fewer Infs because the boundary value is at the very top of the fp32 range so most intermediates saturate to NaN rather than Inf. Same recommendation as K33: align Inductor's LayerNorm lowering with eager's stable variance + rsqrt path.

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

- Issue README: [pytorch-issues/issue173793/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173793/README.md)
- Reproducer: [pytorch-issues/issue173793/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173793/data/repro.py)
- Captured trace: [pytorch-issues/issue173793/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173793/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue173793/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173793/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
