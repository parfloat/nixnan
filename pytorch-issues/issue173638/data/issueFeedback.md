# Feedback for pytorch/pytorch#173638 — K3 — `torch.linalg.slogdet` does not propagate NaN in CUDA

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k3_slogdet_nan_propagation/` on the
`main` branch.

## What the trace observed

- **Verdict:** **HARMFUL-PROPAGATED**
- **Exception summary:** **fp32** NaN=336 ±Inf=108; mem-NaN fp32=1
- **Top kernels launched:** `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `xxtrf4_set_info_ker`, `void getrf_pivot`
- **Kernels emitting events:** `void at::native::elementwise_kernel`, `void at::native::unrolled_elementwise_kernel`, `void at::native::vectorized_elementwise_kernel`
- **Final tensor / printed state: NaN**

## Recommendation (tentative)

The trace shows **336 fp32 arithmetic-NaN events plus 1 fp32 memory-NaN event** from the cuSOLVER LU path. The single memory-NaN is direct evidence the NaN row was **loaded** by a downstream kernel — confirming propagation rather than de-novo NaN generation. The matching fix already exists on CPU; porting that NaN-aware row-selection logic to the CUDA LU pivot would close the gap.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k3_slogdet_nan_propagation/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
