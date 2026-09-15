# Nixnan: GPU Floating-Point Exception Detection

**For comprehensive documentation, see [Tutorial.md](Tutorial.md)**

Nixnan is a binary instrumentation tool for detecting floating-point exceptional values (NaN, Infinity, Subnormals, Division-by-Zero) in NVIDIA CUDA programs. It provides runtime detection without requiring source code modification or recompilation.

## NEW: `bin/autonixnan` — One-Command Automated Triage

`bin/autonixnan` is a Python 3 driver that runs nixnan for you in three phases and
prints one consolidated report. Use it as the *first* thing you try on an unfamiliar
CUDA program: it discovers which kernels exist, finds which of them produce extreme
floating-point magnitudes, and then re-runs the worst offender under full
instrumentation — without you hand-writing a bin specification file or choosing
environment variables.

### Invocation

```bash
bin/autonixnan [-t SECONDS] [-m MAX_REPORTS] -- PROGRAM [ARGS...]
```

The `--` separator is required; everything after it is the target program and its
arguments, invoked exactly as you would run it normally.

```bash
# Simplest form
./bin/autonixnan -- ./my_cuda_program

# A PyTorch workload, capped at 120 seconds per phase
./bin/autonixnan -t 120 -- python train.py --epochs 1

# Allow more extreme-value reports per kernel before the scan self-terminates
./bin/autonixnan -t 600 -m 128 -- ./rd_nixnan
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-t`, `--timeout` | 300 | Wall-clock limit in seconds for the target program in each phase |
| `-m`, `--max-reports` | 32 | Maximum extreme-value reports per kernel before the scan stops itself |

**Requirements:** Python 3.9+ and a built `nixnan.so` in the repository root (run
`make`). The script resolves the library as `<script dir>/../nixnan.so`, so invoke
it from a checkout you have built.

### What it does

1. **Kernel inventory (uninstrumented).** Runs the target with `LOG_KERNELS` pointed
   at a temporary log. Instrumentation is off, so this phase runs at roughly native
   speed. It reports every unique kernel with its call count and total/average
   execution time, plus the full kernel call sequence in invocation order.
2. **Extreme-exponent scan.** Synthesizes a `BIN_SPEC_FILE` on the fly covering the
   *bottom 5%* and *top 5%* of the valid exponent range of each format (`f16`,
   `bf16`, `f32`, `f64`) — the near-underflow and near-overflow binades — with a bin
   count threshold of 1, so the very first value landing in an extreme binade is
   reported. Exception instrumentation is disabled (`INSTRUMENT_EXCEPTIONS=0`) to
   keep this phase cheap, and `NIXNAN_TIMEOUT` bounds the run from inside the
   instrumented process. Functions are then ranked by how many extreme-value reports
   they produced, and the top 10 are printed.
3. **Focused deep run.** The single highest-ranked function is written to a temporary
   `FUNCTION_WHITELIST`, and the program is re-run with instrumentation restricted to
   that one kernel, its output passed straight through to your console.

The target process ending early is normal in phases 2 and 3: it may hit
`NIXNAN_TIMEOUT`, hit the `-m` report cap (nixnan terminates itself), or simply run to
completion. A non-zero exit from those phases is therefore not treated as an error.

Everything `autonixnan` does can also be done by hand with the environment variables
documented below; it just picks sensible defaults for you.

## Quick Start

### Requirements
- **OS**: Linux on x86_64
- **CUDA**: Version 12.x
- **GPU Compute Capability**: ≥ 8.6 (Ampere or newer recommended)
- **Build Tools**: GCC, Make

### Installation

```bash
git clone https://github.com/parfloat/nixnan.git
cd nixnan
make
```

This produces `nixnan.so` in the root directory (the instrumentation library).

### Basic Usage

```bash
# Simple exception detection
LD_PRELOAD=/path/to/nixnan.so ./your_cuda_program

# With line information (compile with -lineinfo)
LINE_INFO=1 LD_PRELOAD=/path/to/nixnan.so ./your_cuda_program

# Log to file instead of stderr
LOGFILE=/tmp/analysis.log LD_PRELOAD=/path/to/nixnan.so ./your_cuda_program
```

## Key Features

### 1. Exception Detection
Detects and reports:
- **NaN** (Not-a-Number) - invalid operations
- **Infinity** - overflow or division by zero  
- **Subnormal** - underflow, precision loss
- **Division by Zero** - explicit divide-by-zero

Supports FP16, BF16, FP32, and FP64 precision levels.

### 2. Binade Monitoring (Exponent Range Tracking)
Monitor floating-point values in specific exponent ranges (binades) to analyze overflow/underflow behavior across different magnitude scales.

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./program
```

Create `spec.json`:
```json
{
    "count": 256,
    "doublings": 7,
    "f16":  [[13, 15]],
    "f32":  [[120, 127]],
    "bf16": [[120, 127]],
    "f64":  []
}
```

**Key parameters:**
- `count`: Report threshold (how many occurrences before reporting)
- `doublings`: Enable adaptive threshold scaling (256→512→1024→...→32768→reset)
- Format arrays: Exponent ranges to monitor `[[min_exp, max_exp]]`

### 3. Adaptive Threshold Doubling
Automatically scale reporting thresholds during execution to observe exception behavior at multiple scales. When a threshold is reached, it doubles (e.g., 256 → 512 → 1024) and resets after N doublings, providing multi-scale insight without data saturation.

### 4. Kernel-Specific Analysis
Track which kernels generate exceptions with per-kernel reporting:
```
#nixnan: f32 bin has reached threshold: kernel=rd_step_fp32 range=[120,127] count=256
```

### 5. Sampling
Reduce overhead for long-running programs by instrumenting only every Nth kernel invocation:

```bash
SAMPLING=64 LD_PRELOAD=./nixnan.so ./program
```

### 6. Memory Instrumentation
Track exceptional values flowing into GPU memory:

```bash
INSTR_MEM=1 LD_PRELOAD=./nixnan.so ./program
```

## Example: Reaction-Diffusion Simulation

The `rd_nixnan.cu` example demonstrates precision comparison across FP16, BF16, and FP32:

```bash
# Compile
nvcc -arch=sm_86 -lineinfo rd_nixnan.cu -o rd_nixnan

# Run with binade tracking and adaptive doubling
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=2 LOGFILE=./analysis.log \
  LD_PRELOAD=./nixnan.so ./rd_nixnan
```

This shows:
- **FP16** overflows at step ~300 (values exceed 65504)
- **BF16** and **FP32** overflow at step ~1900 (values exceed ~3.4e38)
- Exception frequency growth captured by adaptive doubling (256, 512, 1024, ...)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HISTOGRAM` | 0 | Enable exponent range tracking |
| `BIN_SPEC_FILE` | (none) | JSON specification for targeted binade monitoring with adaptive doubling |
| `SAMPLING` | 0 | Instrument every Nth kernel invocation |
| `LOGFILE` | (stderr) | Path to output log file |
| `TOOL_VERBOSE` | 0 | Enable detailed instrumentation logs |
| `LINE_INFO` | 1 | Include source line information (requires -lineinfo compilation flag) |
| `INSTR_MEM` | 0 | Monitor memory operations for exceptions |

## Documentation

**For comprehensive documentation, examples, troubleshooting, and advanced usage, see [Tutorial.md](Tutorial.md)**

The tutorial covers:
- Complete feature explanations
- Detailed binade theory and selection guidelines
- Adaptive threshold doubling mechanics
- Case studies and debugging workflows
- Performance considerations
- Troubleshooting guide

## Publications

- **GPU-FPX**: Li, X., Laguna, I., Fang, B., Swirydowicz, K., Li, A., & Gopalakrishnan, G. (2023). "Design and Evaluation of GPU-FPX: A Low-Overhead tool for Floating-Point Exception Detection in NVIDIA GPUs." *HPDC '23*. https://doi.org/10.1145/3588195.3592991

- **NVBit**: Villa, O., Stephenson, M., Nellans, D., & Keckler, S. W. (2019). "NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs." *MICRO '19*.

## License

See LICENSE file for details.

## Contributing

For issues, questions, or contributions, please open an issue on GitHub or contact the maintainers.

---

**Last Updated**: 2026-08-24  
**Documentation Version**: Complete with binade monitoring and adaptive doubling features
