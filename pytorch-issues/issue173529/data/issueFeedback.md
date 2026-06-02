# Feedback for pytorch/pytorch#173529 — K6 — `nn.Conv2d` produces NaN on CUDA, finite on CPU, near `FLT_MAX` input

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
[pytorch-issues/issue173529/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173529/README.md)
on the `pytorch-issues` branch.

## What the trace observed

- **Verdict:** **INCONCLUSIVE**
- **Exception summary:** _log truncated past trailing report block — no aggregate available_
- **Top kernels launched:** `void cudnn::winograd::generateWinogradTilesKernel`, `cudnn_infer_ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_t…`
- **Kernels emitting events:** `cudnn_infer_ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_t…`, `#nixnan: f32`, `cudnn_infer_ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_t…`
- **Final tensor / printed state: NaN, Inf, CPU/CUDA divergence**

## Recommendation (tentative)

Trace truncated past the 20 MB natural ceiling, so the trailing report block was lost — we can see the per-event SASS for the first ~2 MB but not the aggregate count. The bug definitely fires (output has NaN/Inf in the printed stats); the cuDNN convolution kernel is the source. To get a clean summary we would need a second pass with `PRINT_ILL_INSTR=0` to drop the per-event lines and let the summary block survive inside 20 MB.

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

- Issue README: [pytorch-issues/issue173529/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173529/README.md)
- Reproducer: [pytorch-issues/issue173529/data/repro.py](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173529/data/repro.py)
- Captured trace: [pytorch-issues/issue173529/data/nixnan.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173529/data/nixnan.nnlog)
- Captured stdout: [pytorch-issues/issue173529/data/stdout.nnlog](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/issue173529/data/stdout.nnlog)
- Curated issue index: [pytorch-issues/README.md](https://github.com/parfloat/nixnan/blob/pytorch-issues/pytorch-issues/README.md)
