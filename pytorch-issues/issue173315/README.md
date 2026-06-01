# issue173315 — K26 — `nn.LSTM` CUDA eager — random init proxy

Upstream: https://github.com/pytorch/pytorch/issues/173315

## What this folder is

One of a small set of PyTorch FP-exception issues we ran under
[NixNan](https://github.com/parfloat/nixnan/) for diagnostic value.
See `data/issueFeedback.md` for our reading of the trace and
`usingNixNan/reproduce.sh` to re-run it on your own machine.

## What we did

1. Wrote a minimal CUDA reproducer (`data/repro.py`).
2. Ran it under NixNan with the canonical sweep profile
   (`SAMPLING=1`, per-binade histogram with `count=1024`,
   `ENABLE_FUN_DETAIL=1`, `PRINT_ILL_INSTR=1`, `INSTR_MEM=1`,
   `LINE_INFO=1`). PyTorch 2.3.1+cu121 on RTX 3090 (sm_86).
3. Captured the per-event SASS lines, kernel attribution, and the
   trailing per-precision report block into `data/nixnan.nnlog`.
4. Captured the Python output (CPU value, CUDA value, expected
   behaviour) into `data/stdout.nnlog`.
5. Wrote our reading in `data/issueFeedback.md` — a NixNan-based
   verdict, tentative recommendation, and a few modest questions
   back to the original poster.

## How to reproduce on your machine

```bash
./usingNixNan/reproduce.sh
```

The script will:

1. Build NixNan from this repo (first run, one-time; needs CUDA
   toolkit + GCC).
2. Run `data/repro.py` under NixNan instrumentation.
3. Write fresh logs to `data/nixnan.nnlog.fresh` and
   `data/stdout.nnlog.fresh` so you can compare with the bundled
   originals.

## Requirements

- Linux x86_64.
- NVIDIA GPU, compute capability ≥ 8.6 (Ampere or newer).
- CUDA 12.x toolchain (`nvcc`, headers, runtime libraries).
- GCC + Make.
- Python 3.10+ with `torch` (e.g. `pip install torch`).

If `nvidia-smi` fails with "Driver/library version mismatch", you
need a libcuda whose version matches the loaded kernel module — the
top-level NixNan [Tutorial.md](../../Tutorial.md) describes how to
extract a user-local copy with no root privileges required.

## Files in this folder

```
data/
  repro.py            minimal Python reproducer (runs as a script)
  _diag.py            shared diagnostic helper imported by repro.py
  bin_spec.json       per-binade histogram spec (every fp16/bf16/fp32/fp64 exponent)
  nixnan.nnlog        our captured NixNan trace
  stdout.nnlog        our captured Python output (CPU vs CUDA)
  issueFeedback.md    our reading of the trace + recommendation + questions

usingNixNan/
  reproduce.sh        install + run script for your machine

README.md             this file
```

## Broader context

This bundle is one of many we generated; the full corpus (50 repros,
the rolling synopsis, sweep history, plan documents) lives in
[parfloat/parfloat-class](https://github.com/parfloat/parfloat-class)
under `pytorch-nixnan/`. The `kernel_summary.md` there ranks every
repro by total IEEE-exception event count and includes a
HARMFUL / HARMLESS verdict column.
