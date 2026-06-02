# Feedback for pytorch/pytorch#176415 — K21 — batched `torch.logdet` on 4D float32

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue176415/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176415/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **SILENT-OUTSIDE-FP**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::unrolled_elementwise_kernel<at::native::di…`, `void at::native::reduce_kernel<512, 1, at::native::ReduceOp…`
- **Kernels emitting events:** _no exception events_
- **Final tensor / printed state: NaN**

## Recommendation (tentative)

Our random batch of 4×4×2×2 matrices happened to have positive determinants on our seed; `torch.logdet` ran cleanly with **zero events**. To reproduce the upstream behaviour reliably, the test would need a batch seeded against matrices that include non-positive determinants. Alternatively, providing the explicit input tensor in the upstream issue (rather than relying on `np.random.seed(42)` reproducing across NumPy versions) would make this exact-reproducible.

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

- Issue README: [pytorch-issues/issue176415/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176415/README.md)
- Reproducer: [pytorch-issues/issue176415/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176415/data/repro.py)
- Captured trace: [pytorch-issues/issue176415/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176415/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue176415/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue176415/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
