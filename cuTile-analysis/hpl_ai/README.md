# HPL-AI cuTile Extraction

This folder contains a small cuTile-style Python port extracted from the ICL
HPL-AI reference implementation at:

<https://bitbucket.org/icl/hpl-ai/src/main/>

The upstream reference implementation is BSD-style licensed; see
`HPL_AI_LICENSE`.

The extracted path is the simplest useful numerical core:

- HPL-AI matrix and vector generation
- double-to-single conversion
- single-precision LU factorization without pivoting
- single-precision triangular solve
- single-to-double result conversion
- host-side GMRES refinement in double precision
- HPL-AI scaled residual check

The cuTile kernels cover the compact GPU workload seed: conversion, no-pivot
LU, and triangular solve. GMRES refinement is implemented on the host with
NumPy so the numerical result follows the reference benchmark while the cuTile
surface stays small and inspectable.

The full porting notes are in `../cuTileGenDoc.md`.

## Run

Prepare the local cuTile environment:

```bash
./setup_cutile_venv.sh
source ./cutile_env.sh
```

CPU reference path:

```bash
python hpl_ai/cutile_hpl_ai.py --backend cpu --n 32
```

cuTile path, when `cupy`, `cuda.tile`, and a working NVIDIA driver are
available:

```bash
python hpl_ai/cutile_hpl_ai.py --backend cutile --n 32 --compare-cpu
```

On this workstation, `cutile_env.sh` points the process at the NVIDIA 580.126
driver libraries under `/home/ganesh/opt/nv580.126` and creates a local
`.cutile-libs` shim for the missing `libnvidia-gpucomp.so.580.126.09` soname.
This avoids the system-wide 580.159 user-space library mismatch with the loaded
580.126 kernel module.

Use `--no-gmres` to stop after the extracted mixed-precision LU solve.

Run a multi-configuration sweep from the analysis root:

```bash
./run_cutile_sweep.sh
```

The arrays use the HPL-AI C reference's column-major layout:
`A[row, col]` is stored at `row + col * n`.
