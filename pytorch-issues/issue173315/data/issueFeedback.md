# Feedback for pytorch/pytorch#173315 — K26 — `nn.LSTM` CUDA eager — random init proxy

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k26_lstm_cuda_nan/` on the
`main` branch.

## What the trace observed

- **Verdict:** **INCONCLUSIVE**
- **Exception summary:** _log truncated past trailing report block — no aggregate available_
- **Top kernels launched:** `void at::native::vectorized_elementwise_kernel<4, at::nativ…`, `void at::native::vectorized_elementwise_kernel`, `void at::native::`
- **Kernels emitting events:** `void RNN_blockPersist_fp_RNN_HMMA`, `void RNN_blockPersist_fp_RNN`, `void RNN_blockPersist_fp_LSTM_HMMA`
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

We substituted random initialisation for the upstream's attached `bundle.pt` and `lstm1_weights.pt`. With random inputs the cuDNN LSTM path runs cleanly — **zero NixNan events**. The bug appears to be highly input-dependent; without the specific weights / inputs the NaN never arises. If the attached blobs are still around, re-running the trace against them would tell us whether NixNan can localise the NaN inside the cuDNN RNN kernel.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k26_lstm_cuda_nan/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
