# Feedback for pytorch/pytorch#175156 — K43 — inductor randint inconsistent RNG

A NixNan trace of a minimal reproducer for this issue was captured on
**PyTorch 2.3.1+cu121, RTX 3090 (sm_86)** with the canonical sweep
profile (`SAMPLING=1`, per-binade histogram `count=1024`,
`ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`).

The reproducer, trace, captured stdout, and run command live in
`parfloat/parfloat-class/pytorch-nixnan/repros/k43_inductor_randint_inconsistent/` on the
`main` branch.

## What the trace observed

- **Verdict:** **HARMFUL-LOCAL**
- **Exception summary:** **fp32** sub=2
- **Top kernels launched:** `void at::native::(anonymous namespace)::distribution_elemen…`, `void at::native::unrolled_elementwise_kernel<at::native::di…`, `void at::native::`
- **Kernels emitting events:** `triton__0d1d23c4c5de at /home/ganesh/.local/lib/python3.10/…`
- **Final tensor: clean (no NaN/Inf/divergence in stdout)**

## Recommendation (tentative)

**Zero events** — and even on a torch version where the bug fires, NixNan likely wouldn't surface it: the inconsistency is in the RNG sequence, not in IEEE FP exceptions. NixNan is not a strong tool for RNG-determinism bugs. A deterministic checksum-based test would be more productive.

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

- Reproducer + trace: `parfloat/parfloat-class/pytorch-nixnan/repros/k43_inductor_randint_inconsistent/`
- 50-repro synopsis: `parfloat/parfloat-class/pytorch-nixnan/repros/kernel_summary.md`
- Project narrative: `parfloat/parfloat-class/pytorch-nixnan/May30Summary.md`
