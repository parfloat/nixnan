# Nixnan: GPU Floating-Point Exception Detection

**For comprehensive documentation, see [Tutorial.md](Tutorial.md)**

Nixnan is a binary instrumentation tool for detecting floating-point exceptional values (NaN, Infinity, Subnormals, Division-by-Zero) in NVIDIA CUDA programs. It provides runtime detection without requiring source code modification or recompilation.

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
