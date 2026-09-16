# AutoNX9 (auto-nixnan) Session Summary

Date: 2026-09-16
Branch: `auto-nixnan`
Author: ganesh@cs.utah.edu (assisted by Claude Code)

Summary of the work done in this session exercising `bin/autonixnan`
(the AutoNX9 wrapper around `nixnan.so`) against both a real solver
binary and a purpose-built precision-edge microbenchmark.

## 1. Target under test: `amg_f64_c32` (Prof.g_clamp repo)

Built the mixed-precision AMG solver variants from
`Prof.g_clamp/source_code/amg/amg_mixed_levels.cu` via the project's CMake
build (`clamp_add_precision_policy`), producing four targets:
`amg_policy_64`, `amg_f64_c32`, `amg_f32_c64`, `amg_policy_32`. A helper
script, `source_code/amg/build_and_run_amg_mixed_levels.sh`, configures,
builds, and runs all four with a smoke-sized problem (`--n 12 --cycles
30`); all four converged on the RTX 3090 (sm_86) test machine.

Ran the fp64-fine/fp32-coarse variant under AutoNX9:

```
./bin/autonixnan -t 100 -m 100 -- \
    /home/ganesh/repos/Prof.g_clamp/source_code/build/bin/amg_f64_c32 --n 12 --cycles 30
```

Result: **no extreme exponent values detected** in any FP regime — the
top/bottom-5% exponent scan never triggered, so the full-instrumentation
follow-up pass never ran.

Observation: the "Unique Kernels" section of the report came back empty
(0 kernels) even though the baseline run took ~36.5s vs. ~47ms
uninstrumented — consistent with `nixnan.so` actually attaching and
instrumenting, but writing its kernel-timing log in a format
`autonixnan`'s `collect_kernels()` doesn't parse. `collect_kernels()` sets
`LOG_KERNELS`, while the timing/log machinery elsewhere in the script is
driven by `LOGFILE` + `TIME_KERNELS`. This looks like a pre-existing
mismatch in `bin/autonixnan` itself, not something touched this session,
and it doesn't affect the extreme-value scan (a separate, self-contained
pass).

## 2. Branch housekeeping

Repo was found on `SC26` with uncommitted modifications to
`rd_nixnan`/`traces/Stage2-5/*` that blocked switching to `auto-nixnan`.
Per instruction ("moved many goodies from SC26 to here already"), those
changes were discarded (`git checkout --`) and the repo switched cleanly
to `auto-nixnan`, which tracks `origin/auto-nixnan`.

## 3. `rd_nixnan`: precision-edge reaction-diffusion microbenchmark

`SC26-expts/rd_nixnan/rd_nixnan.cu` solves an explicit FTCS
reaction-diffusion step (`du/dt = D u_xx + lambda u`) in FP16, BF16,
FP32, and FP64, one kernel per format, so nixnan can attribute exceptions
and exponent histograms per precision. The reaction term forces
unbounded exponential growth, driving each format to overflow at a
predictable step count (FP16 ~step 227).

### 3a. Baseline run under AutoNX9

```
./bin/autonixnan -t 1 -m 1 -- ./SC26-expts/rd_nixnan/rd_nixnan
```

The extreme-value scan flags `rd_step_fp16` as the sole function with
extreme exponents, then AutoNX9 automatically re-runs with full
instrumentation whitelisted to that function, streaming live
`#nixnan: error [...] detected ...` lines and finishing with the
underlying `nixnan.so` aggregate report (unique exception sites and
their repeat counts).

### 3b. Larger-domain variant

Created `SC26-expts/rd_nixnan/rd_nixnan_large.cu`: identical kernels and
physics, but `N=1001` (10x the grid points of the original `N=101`) with
`L` scaled up proportionally (`10.0` vs. `1.0`) so `dx`, `r` (diffusion
number), and `lambda*dt` are unchanged — the blow-up timeline is
identical, only the amount of per-step grid data is 10x larger. Built
with `nvcc -arch=sm_86 rd_nixnan_large.cu -o rd_nixnan_large`; baseline
(uninstrumented) run confirmed `r=0.1000`, `lambda*dt=0.0500`, matching
the original exactly.

### 3c. FP16 exception results: `-t 1 -m 1` vs. `-t 100 -m 100`, original vs. 10x domain

`-t`/`-m` had **no effect** on the final exception counts in either
binary — the full-instrumentation whitelisted-function pass isn't capped
by `-m` (that only bounds the separate extreme-value scan) and both
binaries finish in well under a second regardless of `-t`. Both
timeout/report-cap combinations produced byte-identical `nixnan.so`
aggregate reports.

The metric that matters is the aggregate report's **unique exception
sites vs. repeats**:

| | unique NaN sites | NaN repeats | unique Inf sites | Inf repeats | unique Subnormal sites | Subnormal repeats |
|---|---|---|---|---|---|---|
| original (`N=101`) | 22 | 60,534 | 22 | 639 | 0 | 0 |
| large (`N=1001`, 10x) | 22 | **482,828** | 22 | **1,973** | **2** | **110** |

Findings:

- **Unique exception-site counts (22/22) are a property of the source
  code**, not the problem size — they stay fixed at 10x the domain.
- **NaN repeats scale close to linearly with grid size** (~8.0x for a
  10.1x increase in interior grid points, 99 -> 999), consistent with
  the stencil producing NaN at essentially every interior point once the
  reaction term blows up.
- **Infinity repeats scale sub-linearly** (~3.1x for the same 10x
  domain increase) — infinity is reached far less often per point than
  NaN.
- **Subnormal exceptions only appear in the larger domain** (0 -> 2
  unique sites, 110 repeats). The extra near-boundary, low-amplitude
  grid points introduced by the bigger domain are enough to push some
  diffusion increments below the FP16 subnormal floor — a failure mode
  the small `N=101` case never exercises at all.

## Takeaways

1. AutoNX9's `-t`/`-m` flags govern the *extreme-value scan's* time and
   per-kernel report budget, not the final full-instrumentation pass's
   exception accounting — don't expect them to change aggregate
   NaN/Inf/Subnormal counts for a run that finishes quickly on its own.
2. When comparing exception behavior across problem sizes, use the
   `nixnan.so` aggregate report's **repeats**, not the unique-site count
   — the unique-site count is a static property of the code, while
   repeats reflect how much bad floating-point behavior the specific run
   actually produced.
3. Growing a problem's domain size can surface qualitatively new
   exception classes (here, Subnormal) that never trigger at smaller
   scale, in addition to scaling existing ones.
4. The `amg_f64_c32` smoke run showed no extreme exponents at this problem
   size; the `collect_kernels()` kernel-log parsing quirk in
   `bin/autonixnan` (noted in §1) is worth fixing separately so the
   "Unique Kernels" section reports correctly on binaries that don't
   trip the extreme-value scan.

## Next steps (for follow-up session)

- Investigate/fix the `LOG_KERNELS` vs. `LOGFILE`+`TIME_KERNELS`
  mismatch in `bin/autonixnan`'s `collect_kernels()`.
- Consider running the AMG variants at larger problem sizes (`--n`) to
  see whether extreme exponents/FP16 exceptions surface, mirroring the
  `rd_nixnan` domain-scaling result.
- Decide whether `rd_nixnan_large.cu` should be parameterized via `-DN=`
  / a domain-length macro rather than kept as a separate hardcoded file.

---
🤖 Generated with [Claude Code](https://claude.com/claude-code)
