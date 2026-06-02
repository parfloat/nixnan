# Feedback for pytorch/pytorch#176412 — K22 — `torch.erfinv` 2D float32 negative out-of-domain

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue176412/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176412/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMFUL-PROPAGATED**
- **Exception summary:** **fp32** NaN=29
- **Top kernels launched:** `erfinv_kernel_vectorized4_kernel`, `void at::native::vectorized_elementwise_kernel`, `void at::native::unrolled_elementwise_kernel`
- **Kernels emitting events:** `erfinv_kernel_vectorized4_kernel`
- **Final tensor / printed state: NaN**

## Recommendation (tentative)

The trace shows **29 fp32 NaN events** — identical to the 1D sibling #176384 (our K9). Same framework-consistency framing applies: PyTorch's NaN for out-of-domain inputs is defensible, and the divergence with PaddlePaddle's `-inf` is a convention choice, not an IEEE bug.

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

- Issue README: [pytorch-issues/issue176412/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176412/README.md)
- Reproducer: [pytorch-issues/issue176412/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176412/data/repro.py)
- Captured trace: [pytorch-issues/issue176412/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176412/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue176412/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176412/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
