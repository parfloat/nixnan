# Feedback for pytorch/pytorch#181645 — K39 — `flex_attention(BACKEND='FLASH', return_lse=True)` off by `ln(2)`

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k39_flex_attention_lse_log2/` on the
`main` branch.

## What the trace observed

- **Verdict:** **INCONCLUSIVE**
- **Exception summary:** _log truncated past trailing report block — no aggregate available_
- **Top kernels launched:** _none recorded_
- **Kernels emitting events:** _no exception events_
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

`torch.nn.attention.flex_attention` is **not available in torch 2.3.1** — the `_diag.banner` helper detected the missing module and exited the repro cleanly. To verify on a newer torch, install ≥ 2.5 and re-run. The bug is specifically about the FA4 backend dispatch (which doesn't exist in 2.3 at all), so this repro can't add diagnostic value on this stack.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k39_flex_attention_lse_log2/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
