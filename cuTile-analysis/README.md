# cuTile-analysis - NixNan diagnostics for cu-tile kernels

This folder is the starting point for analyzing cu-tile based kernels with
[NixNan](https://github.com/parfloat/nixnan/).

The goal is to collect runnable kernels, captured NixNan traces, and short
notes that explain any observed floating-point exceptional values, exponent
range behavior, or other numerical diagnostics relevant to cu-tile workloads.

## Layout

```
cuTile-analysis/
  README.md
```

Future analyses can add one subdirectory per kernel, benchmark, or issue.

## Current Analyses

- `hpl_ai/`: compact cuTile-style extraction of the ICL HPL-AI reference
  implementation's mixed-precision no-pivot LU solve path.
- `nixnan_sweep/`: NixNan histogram and exception traces for the cuTile solver,
  including a WMMA tensor-core verification probe.

## cuTile Runtime

For the local cuTile GPU environment, run:

```bash
./setup_cutile_venv.sh
source ./cutile_env.sh
```

`cutile_env.sh` points Python and the dynamic linker at `.venv-cutile`, the
matching NVIDIA 580.126 user-space libraries for this workstation, and the
local `.cutile-libs` shim used by cuTile kernel launch.

## NixNan Sweep

To borrow the already-built NixNan from the parent checkout and run the
instrumented tensor-core sweep:

```bash
./link_nixnan.sh
./run_nixnan_cutile_sweep.sh
```

The sweep stores traces under `nixnan_sweep/data`.
