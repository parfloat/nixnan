# testManyFMTs - Multi-Format Floating-Point Exercise

A CUDA test program that exercises floating-point operations across multiple precisions (FP16, FP32, FP64) and binades, designed for use with nixnan instrumentation.

## Overview

This test generates 200 random values in the range `[0.0001, 16.0]` for each floating-point format using a log-uniform distribution to ensure even coverage across ~17 binades (2^-13 to 2^4). It then performs arithmetic operations (add, mul, fma, sub) to exercise the GPU's floating-point units.

## Files

| File | Description |
|------|-------------|
| `testManyFMTs.cu` | CUDA source code |
| `Makefile` | Simple build (test only) |
| `MakeTest` | Full build including nixnan.so dependency |
| `simpleExample.sh` | Interactive demo script |
| `README.md` | This file |

## Quick Start

### Interactive Demo
```bash
./simpleExample.sh
```

### Manual Build and Run
```bash
# Build everything (nixnan.so + test)
make -f MakeTest

# Run without instrumentation (baseline)
./testManyFMTs

# Run with nixnan (detect special values)
LD_PRELOAD=../../nixnan.so ./testManyFMTs

# Run with nixnan + histogram collection
HISTOGRAM=1 LD_PRELOAD=../../nixnan.so ./testManyFMTs
```

## What It Tests

- **FP16 (half)**: 200 values, operations via `__hadd`, `__hmul`, `__hfma`, `__hsub`
- **FP32 (float)**: 200 values, operations via `+`, `*`, `fmaf`, `-`
- **FP64 (double)**: 200 values, operations via `+`, `*`, `fma`, `-`

Each value undergoes 5 operations, totaling 3000 FP operations across all formats.

## Output

The program reports:
1. Binade distribution of generated values (how many values in each power-of-2 range)
2. Sample input values and computed results
3. Summary of operations performed

When run with nixnan:
- **Without HISTOGRAM**: Reports any special values detected (NaN, Inf, denormals)
- **With HISTOGRAM=1**: Additionally collects and reports operand/result magnitude histograms

## Customization

Edit `testManyFMTs.cu` to change:
- `NUM_VALUES`: Number of random values per format (default: 200)
- `MIN_VAL`: Minimum value in range (default: 0.0001)
- `MAX_VAL`: Maximum value in range (default: 16.0)

Pass a seed as command-line argument for reproducibility:
```bash
./testManyFMTs 42
```
