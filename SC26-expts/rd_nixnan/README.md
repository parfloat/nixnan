# reaction_diffusion_gpu under nixnan

CUDA port of the FPChecker tutorial's `tutorial/example_2/reaction_diffusion.cpp`
(in the sibling `SC26-Tutorial-NixNan` repo) run under `nixnan.so`, comparing
FP64 vs FP32 dynamic range on the same 1D reaction-diffusion PDE, plus the
exp-binade histogram / exponential `SAMPLING` sweep pattern established in
`../lu_solve_gpu/`, plus two nixnan features not exercised there
(`PRINT_ILL_INSTR`+`MAX_ERRORS`, `INSTR_MEM`).

## Files

- `reaction_diffusion_gpu.cu` -- CUDA solver for
  `du/dt = D*d2u/dx2 + lambda*u` (explicit FTCS, Dirichlet BCs,
  sine-pulse initial condition), with the exact parameters from
  `tutorial/example_2/reaction_diffusion.cpp`: `L=1, T=4, N=101, D=0.01,
  lambda=25, M=80000` (`dt = T/M = 5e-5`). Precision is a compile-time
  macro (`RD_USE_FLOAT`) instead of the CPU tutorial's hand-edited
  `typedef`, so both variants exist as separate binaries.
- `Makefile` -- builds `reaction_diffusion_gpu_fp64` (default, `double`)
  and `reaction_diffusion_gpu_fp32` (`-DRD_USE_FLOAT`).
- `bin_spec.json` + `gen_bin_spec.py` -- `BIN_SPEC_FILE` for exp-binade
  gathering (see below).
- `run_nixnan_experiment.sh` -- runs everything; outputs land under `logs/`.

## Running

```bash
cd ~/repos/nixnan/SC26-expts/rd_nixnan
./run_nixnan_experiment.sh
```

Requires a built `../../nixnan.so`. Override `NIXNAN_SO`, `CYCLES`, or
`SAMPLING_EXE` via environment variables; see the script header.

## Why this reproduces the CPU tutorial's point on the GPU

`lambda=25` amplifies the solution roughly as `exp(lambda*t)`; by `t=4` the
growth factor is `exp(100) ~= 2.7e43`. That is comfortably inside FP64's
range (max ~1.8e308) but past FP32's (max ~3.4e38). Observed:

- **FP64**: completes cleanly, max value grows to `1.70e43` at `t=4`, zero
  exceptions of any kind (`logs/baseline/fp64.log`, `logs/histogram/fp64.log`).
- **FP32**: overflows to Infinity between `t=3.52` (`1.10e38`, still finite)
  and `t=3.60` (the next printed sample). `Infinity - 2*Infinity + Infinity`
  in the diffusion stencil then produces NaN, which propagates through the
  rest of the domain and the rest of the run.

## A silent-failure wrinkle in the GPU port

`reaction_diffusion_gpu.cu`'s `max_reduce_kernel` mirrors the CPU version's
`std::max_element`: it picks the larger of two values via `v > best`, a
comparison that is `false` whenever either side is NaN. Once Inf/NaN enters
the domain, this reduction can no longer find it -- and in this program it
does not even report the surviving `Infinity` values: `logs/baseline/fp32.log`
shows the reported max literally becomes `0.0000000000e+00` from `t=3.60`
onward, not `nan` or `inf`. This also silently defeats the program's own
`std::isfinite(max_val)` check (there is no "first non-finite" message
anywhere in `logs/baseline/fp32.log`'s stderr), even though the underlying
array genuinely does contain Inf/NaN by then -- confirmed independently by
nixnan, which is not looking at the reduction's output at all but at every
FADD/FFMA as it executes: `logs/histogram/fp32.log` and
`logs/memory/fp32.log` both show real NaN/Infinity counts. See
`rd_nixnan_report.tex` for the full discussion; this is presented as-is
(not "fixed") because it is an authentic, and instructive, replication of
the same hazard the CPU tutorial's own `std::max_element`-based
`print_max_value` carries.

## Stage 1: exp-binade histogram gathering

Unlike `lu_solve_gpu` (one precision per build), both FP64 and FP32 are
"present" here across the two binaries that share `bin_spec.json`:

```json
{
    "count": 1000000,
    "doublings": 7,
    "max_reports": 0,
    "f16": [],
    "bf16": [],
    "f32": [[-126, -123], ..., [124, 127]],
    "f64": [[-1022, -1019], ..., [1022, 1023]]
}
```

- **Bucket size 4**, same convention as `../lu_solve_gpu/gen_bin_spec.py`:
  `f32`'s full valid range `[-126, 127]` (64 buckets) and `f64`'s
  `[-1022, 1023]` (512 buckets), both in width-4 bins.
- **`count: 1,000,000`**: this workload runs `M=80000` time steps over ~99
  interior points, roughly 1000x more instrumented FP-op instances than
  `lu_solve_gpu`'s one-shot $20\times20$ solve. `count: 128` (the value
  used there) produced ~950,000 report lines here; `1,000,000` was chosen
  empirically to land back in the same "nice" ~100-150-reports-per-run
  range (see `gen_bin_spec.py` for the exact reasoning).
- `doublings` is carried for schema compatibility only; as established in
  `../lu_solve_gpu/README.md`, this nixnan build does not read it.

## Stage 2: exponential SAMPLING sweep

Same schedule and same harness as `../lu_solve_gpu/run_nixnan_experiment.sh`:
`SAMPLING=1,2,4,8`, 16 runs ("snaps") per level, reset back to 1 --
implemented as 64 separate process launches of `reaction_diffusion_gpu_fp32`
(one snap = one process), since nixnan reads `SAMPLING` once at init. The
same `SAMPLING`-has-no-effect finding documented in
`../lu_solve_gpu/README.md` applies here too (not re-derived in full; see
`rd_nixnan_report.tex` for a spot-check).

## Stage 3: interesting exception settings

Two nixnan features not exercised in the `lu_solve_gpu` experiment:

- **3a -- `PRINT_ILL_INSTR=1 MAX_ERRORS=20`**: prints the offending SASS
  instruction for each exception and terminates after 20 errors. On FP32
  this hits the cap almost immediately after overflow and pinpoints the
  exact source lines (`reaction_diffusion_gpu.cu:64`, the diffusion-term
  subtraction, and `:66`, the state update) as where Infinity first
  appears. On FP64 it never reaches the cap and simply runs to completion.
- **3b -- `INSTR_MEM=1`**: memory instrumentation, confirming NaN values
  are actually written to the `u_next` buffer in GPU global memory (not
  just transient in registers) -- `logs/memory/fp32.log` shows
  `FP32 Memory Operations: NaN: 1 (36928 repeats)`; `logs/memory/fp64.log`
  shows zero.

## Directory layout after running

```
logs/
  baseline/{fp64,fp32}.log
  histogram/{fp64,fp32}.{log,stdout}
  sampling/cycle_01/sampling_{1,2,4,8}/snap_{01..16}.{log,stdout}
  diagnostic/{fp64,fp32}.{log,stdout}
  memory/{fp64,fp32}.{log,stdout}
```
