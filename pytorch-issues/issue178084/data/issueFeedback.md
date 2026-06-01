# Feedback for pytorch/pytorch#178084 — K33 — `torch.compile` LayerNorm NaN

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k33_compile_layer_norm_nan/` on the
`main` branch.

## What the trace observed

- **Verdict:** **HARMFUL-PROPAGATED**
- **Exception summary:** **fp32** NaN=1656 ±Inf=1859 sub=6; mem-NaN fp32=2
- **Top kernels launched:** `triton__0d1d2d3d4e5de`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::reduce_kernel<512, 1, at::native::ReduceOp…`
- **Kernels emitting events:** `void at::native::`, `void at::native::vectorized_elementwise_kernel`, `void at::native::elementwise_kernel`
- **Final tensor / printed state: NaN**

## Recommendation (tentative)

**Reproduces on torch 2.3.1.** The trace shows **1,656 fp32 NaN, 1,859 ±Inf, 6 subnormal, and 2 fp32 memory-NaN events** from the Inductor-emitted LayerNorm kernel. The 2 memory-NaN events are explicit propagation evidence — a downstream kernel **loaded** a NaN that an earlier Inductor-emitted kernel **stored**. Either the variance computation in the Inductor lowering doesn't use Welford's-style stabilisation, or its rsqrt step lacks the same NaN-safety eager has. Patching the Inductor LayerNorm pass to use the same numerically-stable formula as eager would eliminate the entire cascade.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k33_compile_layer_norm_nan/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
