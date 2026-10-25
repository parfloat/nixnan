# test/ — nixnan regression tests

## Goals

nixnan instruments SASS at run time and reports floating-point exceptions (NaN, +Inf, -Inf,
subnormal, division by zero) per precision (FP16, BF16, FP32, FP64), plus NaNs it sees being stored
to global memory (STG). Each test is a tiny CUDA program that feeds known bad operands to one
instruction family; its `.expect` file is the nixnan report it must produce. The suite answers:

1. **Detection** — is the exception reported, in the right precision section?
2. **No false reports** — do ordinary values give all zeros? (`negative/`)
3. **Stable accounting** — are unique and repeat counts unchanged by edits to the tool?
4. **Clean exit** — does the program under `LD_PRELOAD` finish with the same exit code?

The `.expect` files are **regression baselines, not an independent oracle**: they record what nixnan
printed when they were created. A passing run means "unchanged", not "correct".

## Quick start

    make -C test check                 # build what applies to this GPU, run under nixnan, compare
    make -C test run/fp32/div_single   # one test
    make -C test run/fp16              # one directory
    make -C test help                  # all targets

`check` exits non-zero on any FAIL, ERROR or MISSING; skips alone exit 0. Reports of every run are
saved under `results/<platform>/` (see below).

### Make targets (all exercised on an RTX 3090 / CUDA 12.4)

| command | does |
|---|---|
| `make` | build every test that applies to this GPU |
| `make <dir>/<test>` | build one test, e.g. `make fp32/div_single` |
| `make run/<dir>/<test>` | build, run under nixnan, compare, e.g. `make run/fp32/div_single` |
| `make run/<dir>` | same for a directory (or any prefix), e.g. `make run/mma` |
| `make check` | build + run + compare everything |
| `make record` / `make record-platform` | (re)write `<test>.expect` / `<test>.sm<NN>.expect` |
| `make coverage` | which instrumented opcodes do the tests exercise? |
| `make gen` | regenerate `mma/wmma_*.cu` from `mma/gen_wmma_tests.py` |
| `make platform` / `make list` | show detected platform / tests that apply to it |
| `make clean` | remove built executables |

Variables: `NIXNAN_SO=<path>` (default `../nixnan.so`), `ARCH=sm_86` (default `native`),
`NIXNAN_TEST_SM=70` (pretend the GPU is sm_70, to see the skip logic work).
Note: `make run-fp32/div_single` does **not** work (make strips the directory before matching a
pattern with no slash), so the syntax is `run/...`.

## Layout

```
test/
├── Makefile, run.py, coverage.py, helpers.cuh, .gitignore
├── first_test/    hello_nixnan.cu — the tutorial demo, with collected exceptions (own README)
├── fp16/ fp32/ fp64/   one file per operation family (add, mul, fma, div, sqrt, sin, ...)
├── bf16/          arith.cu — BF16 add/mul/fma, scalar and packed (HFMA2.BF16_V2)
├── mma/           basic.cu (FP16 acc) + generated wmma_{f16_f32acc,bf16_f32acc,tf32,f64}.cu
├── memory/        stg_nan.cu — NaN stored through STG for each width/type
├── negative/      clean_ops.cu — ordinary values in every precision + a tensor core: must be all zeros
├── infinities/    inf.cu
└── results/<platform>/   collected output of the last run on that platform (tracked)
```

## How the runner decides what to do (`run.py`)

- **Platform probe.** GPU name, compute capability (`nvidia-smi`), driver, `nvcc` version. Printed at
  the top of every run and used to name `results/sm86_cuda12.4/`.
- **Annotations** in each `.cu`: `// @min_sm 80`, `// @max_sm N`, `// @timeout S`. A test whose
  requirement is not met is reported as `SKIP (needs sm_80, have sm_70)`, and `make` does not even
  build it. Corrected from the old blanket `@min_sm 20`: FP16 tests 53, `tanh` 75, BF16/TF32/FP64
  MMA 80. Tested with `NIXNAN_TEST_SM=70`.
- **Tests are the `*/*.cu` files**, not "every executable found", so a stale binary is never run.
- **Outcomes:** `PASS`, `FAIL` (report differs), `ERROR` (no nixnan report, crash, timeout, or not
  built), `SKIP`, `MISSING` (no `.expect`). A missing report is an ERROR, never a pass. Each of these
  paths was provoked and checked (bad preload library, edited expectation, `@timeout 3` on an
  infinite loop, deleted binary).
- **Expectation lookup:** `<test>.sm<NN>.expect` (e.g. `sqrt_single.sm86.expect`) first, then
  `<test>.expect`. Use the first for results that legitimately differ by GPU.
- **Default is check.** Recording is explicit (`--record`, `--record-platform`); the old runner
  overwrote every `.expect` unless given `--check`.
- **Environment:** `LD_LIBRARY_PATH` is unset for the tests (`--keep-ld-library-path` to keep it),
  per-test timeout (default 120 s), sorted order. Reports are saved even on failure.
- **Collected exceptions:** `results/<platform>/<dir>/<test>.report` (full report),
  `results/<platform>/exceptions.tsv` (one row per non-zero counter: test, section, kind, unique,
  repeats), `platform.json`, and `coverage.txt`.

## Coverage

`make coverage` disassembles every built test (`cuobjdump -sass`) and matches the opcodes against
`src/instructions.json` the way nixnan does (full-string regex). On sm_86 it finds 15 of the 16
table entries exercised. **`DMMA.8x8x4` has no test**: sm_86 emits `DMMA.884` instead, so the
`8x8x4` form presumably belongs to another architecture; I have not found out which. The same file
lists FP opcodes the tests emit that nixnan does not instrument (`FSETP`, `DSETP`, `FSEL`, `F2F`,
`F2FP`, `FCHK`), i.e. the exceptions those instructions could carry are not checked.

## Findings made while building this

- **Tensor cores.** `mma/` now covers HMMA.16816.F16 (`basic`), HMMA.16816.F32, HMMA.16816.F32.BF16,
  HMMA.1684.F32.TF32 and DMMA.884. The RTX 3090 (sm_86) does run `DMMA.884` — the FP64 WMMA test
  produced the correct `inf`, `nan` and subnormal results and nixnan reported them.
- **STG instrumentation is narrower than "NaN stored to memory".** From `src/meminstrumentation.cu`
  and `memory/stg_nan.cu`: nixnan finds the stored register's type from the previous instruction that
  used it, so a NaN that was merely loaded and stored (LDG → STG, no FP op) is **not** reported, and
  8/16-bit stores (`STG.E.U8/U16`) are skipped. Scalar `half` stores are therefore invisible, while
  packed `half2`/`bfloat162` (32-bit `STG.E`) are seen. `stg_nan.cu` pins this behaviour.
- **`INSTR_MEM`.** In this source tree the `instrument_mem` variable set from `INSTR_MEM` is assigned
  in `nvbit_at_init` and I found no other use; STG instrumentation ran with the variable unset.
- **Five FP64 baselines were stale for this GPU.** `fp64/{div,rcp,rcp_approx,rsqrt,sqrt}_single.expect`
  date from 2025-09-25; the tool has changed since. Current output differs in a few counts, and is
  deterministic (identical over repeated runs). In `div_single` the old baseline reports
  "Division by 0: 0" for 1.0/±0.0, so the old numbers look like the under-report. I did not bisect
  the tool history, so the cause is a hypothesis. I recorded sm86-specific baselines
  (`*.sm86.expect`) and left the old files for other GPUs; they will fail there until re-recorded.
- **Removed:** `fp64/fma.expect` (identical duplicate of `fma_single.expect`) and
  `memory/test.expect` (no source; it did not match `infinities/inf` either).

## Not done yet

- FP16 gaps: div, rcp, sqrt; `half2` min/max/compare; conversions (F2F, I2F, F2I), min/max,
  compare/predicate instructions (their opcodes are not in the instrumentation table at all).
- Tool behaviour tests: environment options (`SAMPLING`, `INSTR_BEGIN/END`, `LOGFILE`, `LINE_INFO`,
  kernel lists), several streams or threads, divergent warps, a large grid with many exceptions per
  launch to stress the device-to-host channel.
- Per-instruction assertions (using `PRINT_ILL_INSTR=1` output) instead of only report totals.
- Assertion-style checks ("at least one NaN, no BF16") for tests whose exact counts depend on how the
  compiler expands an instruction (FP64 divide and sqrt), instead of per-GPU baselines.
- Verification on other GPUs: only sm_86 has been run. Pre-sm_80 and Hopper behaviour (and the
  `DMMA.8x8x4` form) are untested; the skip logic was only exercised with `NIXNAN_TEST_SM`.
- CI.
