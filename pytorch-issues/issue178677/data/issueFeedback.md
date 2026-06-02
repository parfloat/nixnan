# Feedback for pytorch/pytorch#178677 — K41 — TransformerEncoder all-masked

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue178677/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178677/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **HARMLESS-STABLE**
- **Exception summary:** _no exception events_
- **Top kernels launched:** `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::(anonymous namespace)::CatArrayBatchedCopy…`
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

**Zero events** on the eager path — either eager raised before producing NaN (the upstream issue describes eager raising `RuntimeError: to_padded_tensor`), or the masked softmax stayed finite via a defensive constant-fold. The upstream divergence is specifically between eager (raises) and torch.compile (silently succeeds), and our reproducer covers only eager. To exercise the torch.compile silent path we'd want a separate `torch.compile`d reproducer.

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

- Issue README: [pytorch-issues/issue178677/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178677/README.md)
- Reproducer: [pytorch-issues/issue178677/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178677/data/repro.py)
- Captured trace: [pytorch-issues/issue178677/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178677/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue178677/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue178677/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
