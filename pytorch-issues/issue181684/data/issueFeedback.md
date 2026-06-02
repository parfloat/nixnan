# Feedback for pytorch/pytorch#181684 — K29 — `addcdiv` fp16 on CUDA

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue181684/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181684/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMFUL-PROPAGATED**
- **Exception summary:** **fp32** NaN=2310 ±Inf=1380 ÷0=210
- **Top kernels launched:** `void at::native::(anonymous namespace)::distribution_elemen…`, `void at::native::`, `void at::native::vectorized_elementwise_kernel`
- **Kernels emitting events:** `void at::native::vectorized_elementwise_kernel`, `void at::native::elementwise_kernel`, `void at::native::unrolled_elementwise_kernel`
- **Final tensor / printed state: Inf**

## Recommendation (tentative)

The trace shows **2,310 fp32 NaN, 1,380 ±Inf, and 210 explicit ÷0 events** from the addcdiv fp16 kernel. The 210 ÷0 events match the injected zero-divisor elements (2 zeros broadcast through the elementwise + reduction path produces 210 events). This is the cleanest demonstration in the corpus that addcdiv lacks zero-guard logic on fp16. A simple `where(c == 0, input, addcdiv(input, b, c, value))` wrapper at the call site, or a NaN-safe divide inside the kernel, would close the DISABLED CI flake.

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

- Issue README: [pytorch-issues/issue181684/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181684/README.md)
- Reproducer: [pytorch-issues/issue181684/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181684/data/repro.py)
- Captured trace: [pytorch-issues/issue181684/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181684/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue181684/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue181684/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
