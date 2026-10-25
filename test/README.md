# test/ — nixnan regression tests

## Goals

nixnan instruments SASS at run time and reports floating-point exceptions (NaN, +Inf, -Inf,
subnormal, division by zero) per precision (FP16, BF16, FP32, FP64) plus NaNs seen on memory
stores. These tests exist to answer, for each instruction family nixnan claims to handle:

1. **Detection.** If a kernel produces exception X with instruction I, does nixnan report X in the
   right precision section?
2. **No false reports.** If it produces none, does nixnan stay at zero?
3. **Stable accounting.** Are the *unique* and *repeat* counts unchanged by edits to the tool
   (`instructions.json`, operand decoding, the recorder, the channel)?
4. **Clean exit.** Does the program under `LD_PRELOAD` still finish with the same exit code?

Each test is a tiny CUDA program that launches one warp and feeds known bad operands to a single
instruction family (usually via inline PTX, so the instruction is chosen, not left to the compiler).
Its `.expect` file is the nixnan report (from the `#nixnan: ---- nixnan Report ----` marker to the
end of stderr) plus a final `Exit Code:` line.

## Layout

```
test/
├── Makefile        builds every */*.cu into an executable next to it (nvcc -lineinfo -arch=native)
├── run.py          runs every executable under LD_PRELOAD=nixnan.so; records or checks .expect
├── helpers.cuh     fill_array_* and half/bit-pattern helpers
├── fp16/ fp32/ fp64/   one file per operation family (add, mul, fma, div, sqrt, sin, ...)
├── mma/basic.cu    FP16 WMMA tile with injected exceptions (tensor-core / HMMA path)
├── infinities/     inf.cu (no .expect yet)
└── memory/         test.expect only (no source)
```

## How to run

    make -C test                              # build
    python3 test/run.py nixnan.so --check     # compare with .expect files
    python3 test/run.py nixnan.so             # NO --check: RE-RECORDS (overwrites) every .expect

Careful: without `--check` the runner **overwrites the expectations** with whatever the current
build prints, so a regression can be silently accepted. Run `git diff test/` afterwards.

## State on the machine where this README was written

RTX 3090 (sm_86), CUDA 12.4, `nixnan.so` built from the `print_histo_instrn` branch. I built all
tests and ran `run.py --check` with `LD_LIBRARY_PATH` unset:

- 25 passed.
- 5 failed, all in `fp64/`: `rsqrt_single`, `sqrt_single`, `rcp_single`, `rcp_approx_single`,
  `div_single`. For `sqrt_single` the FP64 section differs slightly (NaN 23 (105 repeats) vs 22
  (104); Infinity 8 vs 6; -Infinity 5 vs 3; Subnormal 8 vs 7; Division by 0: 1 vs 0). The cause
  is not established. FP64 divide, sqrt and reciprocal are expanded into multi-instruction
  sequences, so a different GPU, driver or CUDA version could plausibly change the SASS and with it
  the counts, but I have not diffed SASS or checked which platform produced the `.expect` files.
- 1 skipped by the runner: `infinities/inf` (no `.expect`).

## Known problems in the current suite

- **Nothing reads the `// @min_sm 20` header comments** that 29 of the `.cu` files carry. They
  are all `20`, which cannot be right for every file (for example, per the PTX ISA I believe
  `tanh.approx.f32` needs sm_75, and bf16 / tf32 / FP64 MMA need sm_80).
- **`.expect` files are not tied to a platform.** Exact counts depend on the SASS the compiler
  emits, which depends on GPU architecture and CUDA version.
- **`run.py` always exits 0**, even when tests fail, and prints only `[PASSED]`/`[ERROR]` lines.
- **A missing report is not a failure** (`No report marker found` is just printed). A nixnan that
  fails to load, or crashes, can look like a pass for tests that expect nothing.
- **Any executable file under `test/` is run**, including stale binaries (an untracked
  `fp64/add_single` was sitting there), and there is no timeout. A hung test hangs the suite.
- **Directory order is filesystem order**, so output order varies between runs.
- **Orphans:** `fp64/fma.expect` (the source is `fma_single.cu`), `memory/test.expect` (no source),
  `infinities/inf.cu` (no expectation).

## Making the suite complete

Coverage today: FP16 add/sub/mul/fma (scalar and `half2` for add/sub/mul), FP32 add, sub, mul,
fma, div, rcp, sqrt, rsqrt, ex2, lg2, sin, cos, tanh, FP64 add, sub, mul, fma, div, rcp, sqrt,
rsqrt, and one FP16 WMMA program. Proposed additions, roughly in priority order:

1. **Derive the list from the tool.** Write a script that reads `src/instructions.json` (the
   opcode/operand table) and prints every opcode/regex nixnan instruments, then
   lists the ones no test exercises. Make "every instrumented opcode has a test" a check, so new
   table entries cannot ship untested.
2. **BF16.** Every current `.expect` has all-zero BF16 sections, so BF16 detection is untested
   outside the HMMA operand-type path. Add `__nv_bfloat16` / `bfloat162` add, mul, fma (sm_80+).
3. **All tensor-core variants.** `mma/basic.cu` covers one of 17 WMMA variants the toolkit offers.
   `examples/mma-examples/extractmmas.py` already enumerates them (FP16/FP32 accumulate, BF16,
   TF32, FP64, and the 16x16x16, 32x8x16, 8x32x16, 16x16x8, 8x8x4 shapes). Extend the generator
   to poison element `[0,0]` the way `basic.cu` does and emit one test per variant, so HMMA
   operand registers and types for D, A, B and C are all checked.
4. **FP16 gaps:** div, rcp, sqrt, and `half2` fma/min/max/compare.
5. **Instruction classes with no test:** conversions (F2F, I2F, F2I), min/max, compare/set
   predicates, `FSEL`, packed forms, and 32-bit-immediate variants (`FADD32I`, `FMUL32I`).
6. **Memory path.** Restore a source for `memory/` and test NaN at `STG`/`ST` for each type,
   including vector stores and shared/local memory.
7. **Negative tests.** For every family, a run with only ordinary values must report zeros.
   Also `-0.0`, signalling and quiet NaN payloads, and exactly-representable boundary values
   (max normal, min normal, max subnormal).
8. **Tool behaviour, not just instructions.** Environment-variable options (for example
   `LINE_INFO`, `TOOL_VERBOSE`, sampling, kernel white/black lists), several kernels, several
   streams or threads, divergent warps, and a large grid with many exceptions per launch to
   exercise the device-to-host channel (the case where a hang or dropped message would show up).
9. **Check what is being counted.** Several tests assert only the totals. Where the specific
   instruction matters (HMMA vs FFMA), also check the per-instruction lines that `TOOL_VERBOSE`
   prints.

## Making the suite bullet-proof: detect the platform, run only what applies

The aim is that `python3 run.py` on any NVIDIA machine does the right thing and says why when it
skips something.

1. **Probe the platform once, up front.** Query the GPU name and compute capability (a tiny CUDA
   program using `cudaGetDeviceProperties`, falling back to `nvidia-smi --query-gpu=compute_cap`),
   the driver and CUDA runtime versions, `nvcc --version`, and the NVBit version nixnan was built
   with. Print them as a header on every run and store them with recorded results.
2. **Make the annotations real.** Parse a header comment in each `.cu` — `@min_sm`, optionally
   `@max_sm` and `@needs fp64_tensor` — and correct the existing `@min_sm 20` values. Features
   that depend on hardware (BF16 and TF32 need sm_80+, FP64 MMA needs an sm_80+ GPU with the
   instruction, `tanh.approx` needs sm_75+) should skip with a visible `SKIPPED (needs sm_80,
   have sm_86)`, never run and fail.
3. **Build for the detected GPU.** `-arch=native` fails on a toolkit that does not know the GPU;
   fall back to `-arch=sm_<cc>`, and refuse to run a test whose build failed. Do not run
   executables just because they exist: run only the ones the Makefile built from a listed `.cu`,
   and delete stale ones with `make clean`.
4. **Key expectations by platform where they legitimately differ.** For example
   `sqrt_single.expect` as the default plus `sqrt_single.sm86.expect` (or a directory per
   `sm_XX`/CUDA major) used when present. Prefer **assertions over exact counts** where the exact
   count only reflects how the compiler expands an instruction, for instance "FP64 section reports
   at least one NaN and no BF16 exceptions", and keep exact-count checks for the tests designed to
   be exact (a single PTX instruction, an HMMA tile).
5. **Distinguish outcomes.** `PASS`, `FAIL` (output differs), `ERROR` (no report marker,
   crash, or timeout — this must not count as a pass), `SKIP` (platform does not apply),
   `MISSING` (no `.expect`). Print a summary, list the failing names, and **exit non-zero** if
   anything failed or errored. Skips alone should exit 0.
6. **Run safely.** Per-test timeout (so a nixnan hang fails one test instead of the suite), sorted
   order, a scrubbed environment (the `LD_LIBRARY_PATH` setting can matter: on this machine I
   have seen nixnan print no report when it was set, so the runner should either unset it or
   report the problem), and a check that `LD_PRELOAD` actually loaded nixnan by requiring the
   report marker.
7. **Separate record from check.** Make `--check` the default and move re-recording to an explicit
   `--record`, which refuses to run on a dirty tree and prints the diff against the old `.expect`.
8. **Run it in CI where there is a GPU,** on at least one pre-sm_80 GPU, one sm_80/86, and one
   Hopper/Ada machine if available, so that platform-specific behaviour gets exercised rather than
   assumed. Until then, record the platform in each `.expect` header so a failure can be traced
   to "different machine" versus "regression".
