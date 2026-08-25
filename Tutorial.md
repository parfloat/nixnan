# Nixnan Tutorial: Comprehensive Guide to GPU Floating-Point Exception Detection
### Authored by Claude (that might lie) with human edits (that could be fallible - tagged [HE])

## Table of Contents

1. [Introduction](#introduction)
2. [Background: Why Floating-Point Exception Detection Matters](#background)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Basic Usage](#basic-usage)
6. [Environment Variables Reference](#environment-variables-reference)
7. [Advanced Features](#advanced-features)
8. [Understanding the Output](#understanding-the-output)
9. [Case Studies and Debugging Workflows](#case-studies)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

## Introduction <a name="introduction"></a>

Nixnan is a binary instrumentation tool for detecting floating-point exceptional values (NaN, Infinity, Subnormals, Division-by-Zero) in NVIDIA CUDA programs. Built on top of NVBit (NVIDIA Binary Instrumentation Tool), nixnan provides runtime detection capabilities without requiring source code modification or recompilation.

### Key Features

- **Binary-level instrumentation**: Works with closed-source CUDA libraries
- **Multiple precision support**: Detects exceptions in FP16, FP32, and FP64 operations
- **Tensor Core support**: Monitors MMA (Matrix Multiply-Accumulate) instructions including HMMA operations
- **Exponent histogram tracking**: Monitors numerical ranges during execution
- **Source line information**: Reports exception locations with file and line numbers (when debug info available)
- **Low overhead modes**: Sampling support for reduced performance impact
- **Exceptions being written into memory**: Reports exceptions flowing into memory via STG ("store global") [HE]

---

## Background: Why Floating-Point Exception Detection Matters <a name="background"></a>

### The Problem

GPUs are now the dominant platform for machine learning and high-performance computing workloads. Unfortunately, NVIDIA GPUs do not have hardware-level exception trap mechanisms. This means:

1. **Silent failures**: Exceptional values (NaN, INF) can propagate through computations undetected
2. **Unreliable results**: Programs may produce normal-looking outputs that are actually corrupted
3. **Difficult debugging**: Without trapping, locating the source of exceptions is extremely challenging
4. **Closed-source barriers**: Many GPU libraries are binary-only, making source-level debugging impossible

### Types of Floating-Point Exceptions

According to IEEE 754, there are five types of floating-point exceptions:

| Exception | Description | Exceptional Value |
|-----------|-------------|-------------------|
| **Invalid Operation** | Mathematically undefined (e.g., sqrt(-1), 0/0) | NaN |
| **Division by Zero** | Non-zero divided by zero | Infinity (INF) |
| **Overflow** | Result exceeds representable range | Infinity (INF) |
| **Underflow** | Result too small to represent normally | Subnormal [HE] |
| **Inexact** | Result requires rounding | Rounded value |

### Why This Matters for ML and HPC

Consider this common scenario in machine learning:

```python
# Uninitialized tensor - carries garbage values
x = torch.FloatTensor(20, 32, 128).cuda()
# This may contain uninitialized values that may propagate, later generating NaNs [HE]
```

Or in numerical algorithms:

```c
// Division without zero-check
const float recipPrecision = 0.5f / eb;  // If eb is subnormal or zero, this couldexplode [HE]
```

Tools like nixnan help identify these issues before they cause training failures or incorrect scientific results.

### How Binary Instrumentation Helps

Unlike source-level analysis, binary instrumentation:

1. **Works on closed-source code**: Libraries like cuBLAS, cuSPARSE, cuDNN
2. **Sees optimized code**: Catches issues introduced by compiler optimizations
3. **Detects precision changes**: Finds when FP64 operations are downgraded to FP32
4. **Monitors actual execution**: Not static analysis - catches runtime-dependent issues

---

## System Requirements <a name="system-requirements"></a>

- **Operating System**: Linux on x86_64
- **CUDA Version**: 12.x or compatible
- **Compute Capability**: >= 8.6 (Ampere or newer recommended)
- **GPU Driver**: Compatible with CUDA 12
- **Build Tools**: GCC, Make

[HE] : removed mention of ARM
---

## Installation <a name="installation"></a>

### Building from Source

```bash
# Clone the repository
git clone https://github.com/parfloat/nixnan.git
cd nixnan

# Build the instrumentation library
make

# This produces nixnan.so in nvbit_release/tools/nixnan/
```

### Verifying the Installation

```bash
# Compile the basic example
cd examples
nvcc -arch=sm_86 -lineinfo basic.cu -o basic [HE: changed compute_86]

# Run with nixnan instrumentation
LD_PRELOAD=../nvbit_release/tools/nixnan/nixnan.so ./basic
```

---

## Basic Usage <a name="basic-usage"></a>

### Running with Nixnan

The simplest way to use nixnan is via `LD_PRELOAD`:

```bash
LD_PRELOAD=/path/to/nixnan.so ./your_cuda_program [args]
```

### Example with a PyTorch Script

```bash
LD_PRELOAD=/path/to/nixnan.so python train.py
```

### Additional material (presentations/projects) [this section is fully HE]

- This is a great source of info covering NixNan + other tools.
  - [Our SC'25 tutorial listed at the top.](https://fpanalysistools.org)
- Ask to be included in more projects in progress - send email to ganeshutah at gmail.
  - Our [Private Github](https://github.com/parfloat/parfloat-class) 
 
### Example Output

```
--- NVBit (NVidia Binary Instrumentation Tool v1.7.2) Loaded ---
Running #nixnan: kernel [ampere_sgemm_32x128_nn] ...
#nixnan LOC-EXCEP INFO: Warning: in kernel [ampere_sgemm_32x128_nn],
  (SUB) found @ /unknown_path in [ampere_sgemm_32x128_nn]:0 [FP32]
#nixnan LOC-EXCEP INFO: in kernel [ampere_sgemm_32x128_nn],
  NaN found @ /source/file.cu:120 [FP32]

------------ Nixnan Report -----------
--- FP16 Operations ---
Total NaN found: 0
Total INF found: 0
Total underflow (subnormal): 0
Total Division by 0: 0
--- FP32 Operations ---
Total NaN found: 2
Total INF found: 1
Total underflow (subnormal): 2
Total Division by 0: 1
--- FP64 Operations ---
Total NaN found: 0
Total INF found: 0
Total underflow (subnormal): 0
Total Division by 0: 0
--- Other Stats ---
Kernels: 4
The total number of exceptions are: 128
```

### First-Run Workflow with Template Generation

When using binade-targeted monitoring, nixnan can auto-generate a template specification file:

**First run (specification file doesn't exist):**
```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Output:**
```
#nixnan: Created template bin specification file at ./spec.json
#nixnan: Exiting now. Please edit the file to specify which exponent ranges to report.
```

**Template file created (spec.json):**
```json
{
    "count": 128,
    "doublings": 2,
    "bf16": [],
    "f16": [],
    "f32": [],
    "f64": []
}
```

**Second step: Edit the file to specify ranges:**
```json
{
    "count": 256,
    "doublings": 7,
    "bf16": [[120, 127]],
    "f16":  [[13, 15]],
    "f32":  [[120, 127]],
    "f64":  []
}
```

**Final run (with proper configuration):**
```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./analysis.log LD_PRELOAD=./nixnan.so ./rd_nixnan
```

This workflow ensures you:
1. Generate a proper template for your specific formats
2. Edit it to monitor the exponent ranges you care about
3. Run the actual analysis with full binade tracking enabled

---

## Environment Variables Reference <a name="environment-variables-reference"></a>

Nixnan's behavior is controlled through environment variables. These are read at initialization using the NVBit `GET_VAR_INT` and `GET_VAR_STR` macros.

### Instrumentation Control

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `INSTR_BEGIN` | Integer | 0 | Beginning of the instruction interval where to apply instrumentation |
| `INSTR_END` | Integer | UINT32_MAX | End of the instruction interval where to apply instrumentation |
| `SAMPLING` | Integer | 0 | Instrument a repeat kernel every SAMPLING times. Set to N to instrument only every Nth kernel invocation (reduces overhead for repeatedly-called kernels). **Note**: This controls kernel invocation sampling, not to be confused with adaptive threshold doubling (see Histogram Features below) |

### Output and Debugging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TOOL_VERBOSE` | Integer | 0 | Enable verbosity inside the tool. Set to 1 for detailed instrumentation logs |
| `ENABLE_FUN_DETAIL` | Integer | 0 | Enable detailed function information for kernel. Shows additional context about instrumented functions |
| `PRINT_ILL_INSTR` | Integer | 0 | Print the instruction which caused the exception. Useful for debugging specific SASS instructions |
| `LINE_INFO` | Integer | 0 | Enable debug information for source code locations. **Warning**: May cause crashes on some programs; set to 0 if you encounter issues |
| `LOGFILE` | String | (stderr) | Path to the optional log file. Default is to print to stderr. Useful when the instrumented program is capturing stderr |

### Memory Instrumentation

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `INSTR_MEM` | Integer | 0 | Instrument memory instructions for NaN/Inf detection. Monitors load/store operations for exceptional values |

### Histogram Features

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HISTOGRAM` | Integer | 0 | Enable whole-program exponent range tracking. Generates reports like "Exponent range for f16: [-5, 3]". Use with `BIN_SPEC_FILE` for binade-level monitoring |
| `BIN_SPEC_FILE` | String | (none) | Path to JSON specification file for binade (exponent range) monitoring with optional adaptive threshold doubling. See "Understanding Binades" section below |

### Usage Examples

**Basic exception detection:**
```bash
# Basic usage with verbose output
TOOL_VERBOSE=1 LD_PRELOAD=./nixnan.so ./my_program

# Enable source line information (compile with -lineinfo)
LINE_INFO=1 LD_PRELOAD=./nixnan.so ./my_program

# Sample every 64th kernel invocation (for long-running programs)
SAMPLING=64 LD_PRELOAD=./nixnan.so ./my_program

# Log to file instead of stderr
LOGFILE=/tmp/nixnan.log LD_PRELOAD=./nixnan.so ./my_program
```

**Binade/Histogram monitoring (using rd_nixnan.cu as example):**
```bash
# Simple histogram: global exponent ranges
HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan

# Binade monitoring: first run generates template
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./rd_nixnan
# Now edit spec.json with your desired ranges

# Binade monitoring with adaptive doubling: detailed multi-scale analysis
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=2 LOGFILE=./analysis.log \
  LD_PRELOAD=./nixnan.so ./rd_nixnan

# Multiple precision comparison: FP16, BF16, FP32 side-by-side
# (rd_nixnan runs all three precisions in one execution)
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LOGFILE=./all_precisions.log \
  LD_PRELOAD=./nixnan.so ./rd_nixnan
```

**Advanced analysis:**
```bash
# Enable memory instrumentation
INSTR_MEM=1 LD_PRELOAD=./nixnan.so ./my_program

# Limit instrumentation to specific instruction range
INSTR_BEGIN=100 INSTR_END=500 LD_PRELOAD=./nixnan.so ./my_program

# Combined: binade tracking, sampling, line info, and logging
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=2 LINE_INFO=1 LOGFILE=./debug.log \
  LD_PRELOAD=./nixnan.so ./rd_nixnan
```

---

## Advanced Features <a name="advanced-features"></a>

### Tensor Core Monitoring

Nixnan supports instrumentation of Tensor Core operations, including:

- **HMMA instructions**: Half-precision Matrix Multiply-Accumulate
- **IMMA instructions**: Integer Matrix Multiply-Accumulate
- **Various formats**: F16, BF16, TF32, F32 accumulation

Example detection output:

```
HMMA.1688.F32.TF32 R4, R132.reuse, R2, R4 ; : MMA being used!
#nixnan LOC-EXCEP INFO: in kernel [void cutlass::Kernel],
  NaN found @ /unknown_path in [void cutlass::Kernel]:0 [FP32]
```

### Exponent Histogram Tracking

#### Whole-Program Mode

```bash
HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./my_program
```

Output:
```
Exponent range for f16: [-5, 3]
Exponent range for f32: [-12, 15]
Exponent range for f64: [-50, 100]
```

This shows the overall range of exponents observed but does not provide binned/bucketed statistics.

#### Binade-Targeted Range Monitoring with Adaptive Doubling

For detailed exception tracking across specific exponent ranges (binades), create a JSON specification file:

```json
{
  "count": 256,
  "doublings": 7,
  "bf16": [[120, 127]],
  "f16":  [[13, 15]],
  "f32":  [[120, 127]],
  "f64":  []
}
```

**Parameters:**
- `count`: Initial threshold for binned reporting (report when reaching 256 occurrences)
- `doublings`: Enable adaptive threshold doubling; threshold will double (256→512→1024→...→32768) up to N times, then reset
- Format arrays: `[[min_exp, max_exp]]` format, where exponents are in the unbiased range

Run with specification:

```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./my_program
```

**Output example:**
```
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=256
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=512
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=1024
...
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=32768
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=256  (resets after doublings limit)
```

See the "Understanding Binades and Adaptive Threshold Doubling" section below for detailed explanation.

### Understanding Binades and Adaptive Threshold Doubling

#### What is a Binade?

In IEEE 754 floating-point arithmetic, a **binade** is a set of numbers with the same exponent. For example, in FP32:
- Binade [120, 127]: All numbers whose exponents fall between 120 and 127
- Binade [13, 15]: Smaller range for FP16

Binades are useful for:
1. **Overflow detection**: Monitoring high exponent ranges (close to infinity)
2. **Underflow detection**: Monitoring low exponent ranges (close to subnormal)
3. **Precision analysis**: Understanding which magnitude ranges are most affected by exceptions
4. **Performance profiling**: Identifying exception hotspots at specific scales

#### Example: Reaction-Diffusion Simulation

The `rd_nixnan.cu` example demonstrates binade monitoring in a reaction-diffusion FTCS solver:

```bash
# Compile
nvcc -arch=sm_86 -lineinfo rd_nixnan.cu -o rd_nixnan

# Create specification for overflow monitoring
cat > spec.json << 'EOF'
{
    "count": 256,
    "doublings": 7,
    "bf16": [[120, 127]],
    "f16":  [[13, 15]],
    "f32":  [[120, 127]],
    "f64":  []
}
EOF

# Run with binade tracking
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=2 LOGFILE=./analysis.log \
  LD_PRELOAD=/path/to/nixnan.so ./rd_nixnan
```

**What happens:**
- The simulation grows values exponentially until overflow occurs
- FP16 overflows at step ~300 (values exceed 65504)
- BF16 overflows at step ~1900 (values exceed ~3.4e38)
- FP32 overflows at step ~1900 (values exceed ~3.4e38)

**Binade output:**
```
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=256
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=512
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=1024
...
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=32768
#nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=256  <- Resets

#nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=256
#nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=512
... (multiple complete doubling cycles)
```

#### How to Choose Binade Ranges

**For FP16 (5-bit exponent, range -14 to 15):**
- Overflow range: `[[13, 15]]` - catches numbers close to 65504
- Underflow range: `[[-14, -10]]` - catches subnormal transitions

**For FP32 (8-bit exponent, range -126 to 127):**
- Overflow range: `[[120, 127]]` - catches numbers close to 3.4e38
- Underflow range: `[[-126, -100]]` - catches subnormal transitions

**For BF16 (8-bit exponent, range -126 to 127):**
- Overflow range: `[[120, 127]]` - same as FP32 range

**For FP64 (11-bit exponent, range -1022 to 1023):**
- Overflow range: `[[1015, 1023]]` - catches numbers close to 1.8e308
- Underflow range: `[[-1022, -900]]` - catches subnormal transitions

#### Adaptive Threshold Doubling

The `doublings` parameter enables **adaptive sampling** at multiple scales:

```json
{
    "count": 256,
    "doublings": 7,
    "f16": [[13, 15]]
}
```

**Behavior:**
1. **Initial phase**: Report when reaching 256 occurrences in range [13,15]
2. **First doubling**: Threshold becomes 512, report at 512 occurrences
3. **Second doubling**: Threshold becomes 1024, report at 1024 occurrences
4. **... continues**: 2048, 4096, 8192, 16384, 32768
5. **After 7 doublings**: Reset to original 256, repeat cycle

**Why use this?**
- **Early detection**: Catch exceptions quickly with lower thresholds
- **Scale-aware**: Observe behavior changes as exception rates grow
- **Automatic adaptation**: No need to manually adjust count between runs
- **Prevention of overflow**: Prevents threshold from growing infinitely large

**Example output pattern:**
```
count=256    <- Initial threshold reached
count=512    <- After 1st doubling
count=1024   <- After 2nd doubling
count=2048   <- After 3rd doubling
count=4096   <- After 4th doubling
count=8192   <- After 5th doubling
count=16384  <- After 6th doubling
count=32768  <- After 7th doubling
count=256    <- RESET, cycle repeats
```

### Kernel-Specific Analysis

Each binade report includes the kernel name that generated the exception:

```
#nixnan: f32 bin has reached threshold: kernel=ampere_sgemm_32x128_nn range=[120,127] count=256
#nixnan: f32 bin has reached threshold: kernel=ampere_sgemm_32x128_nn range=[120,127] count=512
#nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=256
```

**Use cases:**
- Identify which kernels generate exceptions
- Compare exception patterns across different kernels
- Isolate problems to specific library functions (cuBLAS, cuDNN, custom kernels)

In the `rd_nixnan.cu` example, three separate kernels run:
- `rd_step_fp16`: FP16 reaction-diffusion step
- `rd_step_bf16`: BF16 reaction-diffusion step
- `rd_step_fp32`: FP32 reaction-diffusion step

Each kernel's overflow behavior is tracked independently, showing precision-specific characteristics.

### Memory Instrumentation Mode

When `INSTR_MEM=1`, nixnan also monitors memory operations:

```bash
INSTR_MEM=1 LD_PRELOAD=./nixnan.so ./my_program
```

This detects exceptional values being loaded from or stored to GPU memory, helping identify:
- Uninitialized memory containing NaN patterns
- Corrupted data in global memory
- Exception propagation through memory

---

## Understanding the Output <a name="understanding-the-output"></a>

### Exception Location Information

```
#nixnan LOC-EXCEP INFO: in kernel [kernel_name],
  NaN found @ /path/to/source.cu:120 [FP32]
```

Components:
- **kernel_name**: CUDA kernel where exception occurred
- **path/to/source.cu:120**: Source file and line (if compiled with `-lineinfo`)
- **FP32**: Floating-point precision (FP16, FP32, or FP64)

### Final Report Format

```
------------ Nixnan Report -----------
--- FP16 Operations ---
Total NaN found: X
Total INF found: X
Total underflow (subnormal): X
Total Division by 0: X
--- FP32 Operations ---
...
--- FP64 Operations ---
...
--- Other Stats ---
Kernels: N
The total number of exceptions are: M
```

### Severity Assessment

| Exception | Severity | Typical Impact |
|-----------|----------|----------------|
| **NaN** | High | Computation is corrupted; NaN propagates |
| **INF** | High | Overflow occurred; may cascade to NaN |
| **Division by 0** | High | Usually indicates logic error |
| **Subnormal** | Medium | Precision loss; may be flushed to zero |

---

## Case Studies and Debugging Workflows <a name="case-studies"></a>

### Case Study 1: SRU (Simple Recurrent Unit) NaN Issue

**Problem**: NaN values appearing at the output of a PyTorch-based neural network.

**Detection**:
```bash
LD_PRELOAD=./nixnan.so python run_sru.py
```

**Output**:
```
Running #nixnan: kernel [ampere_sgemm_32x128_nn] ...
#nixnan LOC-EXCEP INFO: in kernel [ampere_sgemm_32x128_nn],
  NaN found in [ampere_sgemm_32x128_nn]:0 [FP32]
```

**Root Cause**: The input tensor was created with uninitialized memory:
```python
x = torch.FloatTensor(20, 32, 128).cuda()  # WRONG: uninitialized
```

**Fix**:
```python
x = torch.randn(20, 32, 128).cuda()  # CORRECT: initialized
```

### Case Study 2: Lossy Data Compressor

**Problem**: NaN exceptions in a GPU-based data compressor.

**Detection with line info**:
```bash
LINE_INFO=1 LD_PRELOAD=./nixnan.so ./compressor
```

**Output**:
```
#nixnan LOC-EXCEP INFO: NaN appears at the destination @
/home/user/compressor/main1.cu:120
Instruction: FFMA R3, R4, -R0, 1 ;
```

**Root Cause**: Line 120 contained:
```c
const float recipPrecision = 0.5f / eb;  // eb was subnormal, causing INF
```

**Fix**: Add input validation for the error bound parameter.

### Case Study 3: Reaction-Diffusion Simulation with Precision Comparison (rd_nixnan.cu)

**Problem**: Need to compare floating-point exception behavior across FP16, BF16, and FP32 in a PDE solver.

**Setup**: The `rd_nixnan.cu` example solves a reaction-diffusion equation:
```c
du/dt = D * u_xx + lambda * u
```

With parameters:
- Grid points: N=101, time steps: M=2500
- Diffusion coefficient: D=0.01
- Reaction term: lambda=50.0
- Expected overflow around step 1818 (t≈1.82)

**Specification for binade monitoring** (spec.json):
```json
{
    "count": 256,
    "doublings": 7,
    "bf16": [[120, 127]],
    "f16":  [[13, 15]],
    "f32":  [[120, 127]],
    "f64":  []
}
```

**Run with analysis:**
```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=2 LOGFILE=./analysis.log \
  LD_PRELOAD=/path/to/nixnan.so ./rd_nixnan
```

**Key findings from output:**

1. **FP16 behavior** (5-bit exponent, max ≈ 65504):
   ```
   #nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=256
   #nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=512
   #nixnan: f16 bin has reached threshold: kernel=rd_step_fp16 range=[13,15] count=1024
   ... (rapid doubling cycles due to fast overflow)
   first non-finite at step 300 (t=0.300)   <- Overflows very early
   ```

2. **BF16 behavior** (8-bit exponent, max ≈ 3.4e38):
   ```
   #nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=256
   #nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=512
   ... (multiple complete doubling cycles)
   #nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=32768
   #nixnan: bf16 bin has reached threshold: kernel=rd_step_bf16 range=[120,127] count=256  <- Reset
   first non-finite at step 1900 (t=1.900)
   ```

3. **FP32 behavior** (same range as BF16 but better mantissa):
   ```
   #nixnan: f32 bin has reached threshold: kernel=rd_step_fp32 range=[120,127] count=256
   #nixnan: f32 bin has reached threshold: kernel=rd_step_fp32 range=[120,127] count=512
   ... (similar to BF16)
   first non-finite at step 1900 (t=1.900)   <- Same timing, cleaner mantissa
   ```

**Summary report:**
```
#nixnan: --- FP16 Operations ---
#nixnan: NaN:                   22 (100134 repeats)
#nixnan: Infinity:              22 (639 repeats)

#nixnan: --- BF16 Operations ---
#nixnan: NaN:                   15 (32975 repeats)
#nixnan: Infinity:               6 (76 repeats)

#nixnan: --- FP32 Operations ---
#nixnan: NaN:                   19 (40966 repeats)
#nixnan: Infinity:               8 (90 repeats)
```

**Insights:**
- FP16 is unusable for this problem (overflows at t≈0.3)
- BF16 and FP32 both reach overflow at t≈1.9, as expected
- BF16 has fewer unique exceptions due to lower mantissa precision
- The adaptive doubling (256→512→1024→...→32768) captures the progressive growth of exception frequency
- Multi-kernel tracking shows precision-specific overflow characteristics

**Debugging approach:**
1. First run identified that FP16 fails early
2. Second run with `doublings: 7` showed exception frequency growth patterns
3. Comparison of three kernels revealed precision-dependent behavior
4. Adaptive thresholds prevented data saturation while tracking detailed patterns

### Case Study 4: CUDA GMRES Solver

**Problem**: Residual always NaN from the first iteration.

**Detection**:
```bash
LD_PRELOAD=./nixnan.so ./gmres_solver
```

**Output**:
```
#nixnan LOC-EXCEP INFO: in kernel [csrsv2_solve_upper_nontrans_byLevel_kernel],
  DIV0 found @ /unknown_path:0 [FP64]
#nixnan LOC-EXCEP INFO: in kernel [MassIPTwoVec],
  NaN found @ /home/user/customKernels.cu:31 [FP64]
```

**Root Cause**: Division by zero in LU factorization due to near-singular matrix.

**Fix**: Used cuSparse's matrix diagonal boosting API:
```c
cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
cusparseXcsrilu02_zeroPivot(handle, info, &position);
// Boost small pivots
```

### Debugging Workflow

1. **Initial Detection**:
   ```bash
   LD_PRELOAD=./nixnan.so ./your_program
   ```

2. **Enable Line Information** (recompile with `-lineinfo`):
   ```bash
   nvcc -lineinfo -g your_program.cu -o your_program
   LINE_INFO=1 LD_PRELOAD=./nixnan.so ./your_program
   ```

3. **Identify First Exception**: Look for the first `LOC-EXCEP INFO` message

4. **Analyze Exception Flow**: Check if exceptions:
   - Appear (generated fresh)
   - Propagate (passed through operations)
   - Disappear (masked by operations like FSEL)

5. **Examine Instruction Context**:
   ```bash
   PRINT_ILL_INSTR=1 LD_PRELOAD=./nixnan.so ./your_program
   ```

6. **For Long-Running Programs, Use Sampling**:
   ```bash
   SAMPLING=64 LD_PRELOAD=./nixnan.so ./your_program
   ```

---

## Performance Considerations <a name="performance-considerations"></a>

### Expected Overhead

Binary instrumentation inherently adds overhead. Typical slowdowns:

| Mode | Slowdown | Use Case |
|------|----------|----------|
| Basic detection | 10-50x | Development/debugging |
| With line info | 20-100x | Detailed debugging |
| With sampling=64 | 2-10x | Long-running programs |
| Memory instrumentation | 50-200x | Deep analysis |

### Reducing Overhead

1. **Use Sampling for Repeated Kernels**:
   ```bash
   SAMPLING=256 LD_PRELOAD=./nixnan.so ./my_program
   ```
   This instruments only every 256th invocation of a kernel.

2. **Limit Instruction Range**:
   ```bash
   INSTR_BEGIN=1000 INSTR_END=2000 LD_PRELOAD=./nixnan.so ./my_program
   ```

3. **Disable Line Info** (if causing issues):
   ```bash
   LINE_INFO=0 LD_PRELOAD=./nixnan.so ./my_program
   ```

4. **Two-Phase Approach**:
   - First run: Fast detection to identify problematic kernels
   - Second run: Detailed analysis on specific kernels

### Performance Data (from GPU-FPX paper)

On a benchmark of 151 HPC and ML programs:
- Over 60% experienced less than 10x slowdown
- Sampling with factor 64 reduced geometric mean slowdown to ~5x
- Compared to BinFPE: 16x faster geometric-mean runtime

---

## Troubleshooting <a name="troubleshooting"></a>

### Common Issues

#### 1. Crashes with LINE_INFO=1

**Symptom**: Program crashes when enabling source line information.

**Solution**:
```bash
LINE_INFO=0 LD_PRELOAD=./nixnan.so ./my_program
```

The line info feature may not work with all programs. Use without it for initial detection.

#### 2. "/unknown_path" in Output

**Symptom**: Exception locations show `/unknown_path` instead of source files.

**Solution**: Recompile your CUDA code with debug information:
```bash
nvcc -lineinfo -g your_program.cu -o your_program
```

#### 3. NVBit Version Mismatch

**Symptom**: Tool fails to load or produces errors about NVBit version.

**Solution**: Ensure your CUDA driver and NVBit versions are compatible. Check:
```bash
nvidia-smi  # Check driver version
nvcc --version  # Check CUDA toolkit version
```

#### 4. Missing Exceptions in Closed-Source Libraries

**Symptom**: Exceptions detected but no source location available.

**Explanation**: For closed-source libraries (cuBLAS, cuDNN, etc.), source information is unavailable. The tool still detects exceptions but can only report kernel names.

**Workaround**: Use the kernel name to identify which library function is causing issues, then check your inputs to that function.

#### 5. Very High Overhead

**Symptom**: Program runs extremely slowly.

**Solution**: Use sampling:
```bash
SAMPLING=128 LD_PRELOAD=./nixnan.so ./my_program
```

#### 6. Output Mixed with Program Output

**Symptom**: Nixnan output interferes with program output.

**Solution**: Redirect nixnan output to a file:
```bash
LOGFILE=/tmp/nixnan.log LD_PRELOAD=./nixnan.so ./my_program
```

#### 7. Binade Threshold Overflow Warning

**Symptom**: Error message about threshold overflow when using large `doublings` parameter:
```
Doubling count threshold of X by Y times would cause overflow. 
Please decrease count threshold or number of doublings.
Exiting now.
```

**Explanation**: The sum `(bit_width(count) + doublings)` must fit in 64 bits. Large `count` values (near 2^63) cannot be doubled many times.

**Solution**: Use smaller `doublings` value or smaller `count`:
```json
{
    "count": 256,
    "doublings": 7,    <- Instead of 20
    "f16": [[13, 15]]
}
```

Example calculations:
- `count: 256, doublings: 7` → OK (256 = 2^8, can double 7 times safely)
- `count: 1024, doublings: 30` → ERROR (1024 = 2^10, can only double ~53 times before overflow)
- `count: 1, doublings: 63` → OK (1 = 2^0, can double up to 63 times)

#### 8. Missing Binade Output in Log

**Symptom**: Expected binade threshold messages don't appear in log file.

**Likely causes:**
1. Exception frequency is lower than `count` threshold - no thresholds reached
2. Exponent ranges don't match where exceptions actually occur
3. Kernel invocation sampling (`SAMPLING` parameter) skipped the exceptions

**Solution**: 
- Start with lower `count` value (e.g., 10 instead of 256)
- Verify your binade ranges match the problem area
- Disable `SAMPLING` for initial analysis
```bash
BIN_SPEC_FILE=./spec.json HISTOGRAM=1 SAMPLING=0 LOGFILE=./test.log \
  LD_PRELOAD=./nixnan.so ./my_program
```

---

## References <a name="references"></a>

### Papers

1. **GPU-FPX Paper**: Li, X., Laguna, I., Fang, B., Swirydowicz, K., Li, A., & Gopalakrishnan, G. (2023). "Design and Evaluation of GPU-FPX: A Low-Overhead tool for Floating-Point Exception Detection in NVIDIA GPUs." *HPDC '23*. https://doi.org/10.1145/3588195.3592991

2. **Array Programming Paper**: Li, X., Baranowski, M., Dam, H., & Gopalakrishnan, G. (2025). "Array Programming on GPUs: Challenges and Opportunities." *ARRAY '25*. https://doi.org/10.1145/3736112.3736144

3. **NVBit**: Villa, O., Stephenson, M., Nellans, D., & Keckler, S. W. (2019). "NVBit: A Dynamic Binary Instrumentation Framework for NVIDIA GPUs." *MICRO '19*.

### Related Tools

- **GPU-FPX**: https://github.com/LLNL/GPU-FPX
- **FPChecker**: LLVM-based exception detection for Clang-compiled CUDA
- **BinFPE**: Earlier SASS-level binary instrumentation tool
- **FloatGuard**: Exception detection for AMD GPUs

### IEEE Standards

- IEEE 754-2008: Standard for Floating-Point Arithmetic
- IEEE 754-2019: Latest revision with updated NaN handling

### Useful Resources

- NVIDIA CUDA Floating-Point Documentation: https://docs.nvidia.com/cuda/floating-point/
- IEEE-754 Floating Point Converter: https://www.h-schmidt.net/FloatConverter/IEEE754.html

---

## Appendix: SASS Instruction Reference

Nixnan instruments the following SASS floating-point instructions:

### Computation Opcodes

| Instruction | Description |
|-------------|-------------|
| FADD | FP32 Add |
| FADD32I | FP32 Add (immediate) |
| FFMA | FP32 Fused Multiply and Add |
| FFMA32I | FP32 Fused Multiply and Add (immediate) |
| FMUL | FP32 Multiply |
| FMUL32I | FP32 Multiply (immediate) |
| MUFU | FP32 Multi Function Operation (sin, cos, sqrt, rcp, etc.) |
| DADD | FP64 Add |
| DFMA | FP64 Fused Multiply Add |
| DMUL | FP64 Multiply |

### Control Flow Opcodes

| Instruction | Description |
|-------------|-------------|
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FMNMX | FP32 Minimum/Maximum |
| DSETP | FP64 Compare And Set Predicate |

### Tensor Core Instructions

| Instruction | Description |
|-------------|-------------|
| HMMA | Half-precision Matrix Multiply-Accumulate |
| IMMA | Integer Matrix Multiply-Accumulate |

---

*This tutorial is part of the nixnan project. For the latest updates, visit: https://github.com/parfloat/nixnan*
