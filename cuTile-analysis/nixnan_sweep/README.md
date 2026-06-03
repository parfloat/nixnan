# NixNan cuTile Sweep

This directory stores NixNan traces for the cuTile HPL-AI extraction.

## Borrowed NixNan

The analysis intentionally borrows the already-built NixNan from one directory
up instead of reinstalling it:

```bash
./link_nixnan.sh
```

This creates:

- `borrowed-nixnan -> ../nvbit_release/tools/nixnan`
- `nixnan.so -> borrowed-nixnan/nixnan.so`

`run_nixnan_cutile_sweep.sh` refreshes those symlinks before each run.

## Run

From `cuTile-analysis`:

```bash
./run_nixnan_cutile_sweep.sh
```

The default sweep runs:

- `MATRIX_SIZES="4"`
- `MATRIX_KINDS="hpl-ai conditioned"`
- `CONDITION_NUMBERS="10"` for conditioned matrices
- `SOLVER_PRECISION_CONFIGS="float64,float32,float32,float64,float64 float64,float16,float16,float64,float64"`

Each solver precision tuple is:

```text
input,factor,solve,refinement,residual
```

The NixNan settings are:

- `SAMPLING=0`, so repeat kernels are not sampled
- `HISTOGRAM=1`
- `BIN_SPEC_FILE=nixnan_sweep/data/bin_spec.json`
- histogram `count=1024`
- `INSTR_MEM=1`
- `PRINT_ILL_INSTR=1`

## Tensor Cores

The sweep enables `TENSOR_CORE_PROBE=1` by default. The probe is a small
FP16xFP16->FP32 WMMA kernel using `wmma::mma_sync`; this is separate from the
cuTile LU solver and is included to force tensor-core SASS during each NixNan
run.

Before running solver cases, the script runs a focused verbose verification and
requires `HMMA` to appear in:

```text
nixnan_sweep/data/tensor_core_verify/nixnan.verbose.nnlog
```

## Output Layout

```text
nixnan_sweep/data/
  bin_spec.json
  summary.tsv
  tensor_core_verify/
    stdout.nnlog
    nixnan.verbose.nnlog
  n4_.../
    run.env
    stdout.nnlog
    nixnan.nnlog
```

Each case directory mirrors the previous `pytorch-issues/issueNNN/data` pattern:
`stdout.nnlog` captures process stdout/stderr, `nixnan.nnlog` captures the
NixNan trace, and `run.env` records the exact case configuration.
