# Feedback for pytorch/pytorch#173793 — K34 — LayerNorm `torch.compile` near `1e37`

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k34_compile_layer_norm_large/` on the
`main` branch.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k34_compile_layer_norm_large/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
