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

### Our SC'25 Tutorial [this section is fully HE]

[https://fpanalysistools.org/](This is a great source of info covering NixNan + other tools.)

(Drill into SC25 at the top.)

### Private Github

[https://github.com/parfloat/parfloat-class](Ask to be included in more projects in progress - send email to ganeshutah at gmail)
 
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

---

## Environment Variables Reference <a name="environment-variables-reference"></a>

Nixnan's behavior is controlled through environment variables. These are read at initialization using the NVBit `GET_VAR_INT` and `GET_VAR_STR` macros.

### Instrumentation Control

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `INSTR_BEGIN` | Integer | 0 | Beginning of the instruction interval where to apply instrumentation |
| `INSTR_END` | Integer | UINT32_MAX | End of the instruction interval where to apply instrumentation |
| `SAMPLING` | Integer | 0 | Instrument a repeat kernel every SAMPLING times. Set to N to instrument only every Nth kernel invocation (reduces overhead for repeatedly-called kernels) |

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
| `HISTOGRAM` | Integer | 0 | Enable whole-program exponent range tracking. Generates reports like "Exponent range for f16: [-5, 3]" |
| `BIN_SPEC_FILE` | String | (none) | Path to JSON specification file for targeted range monitoring |

### Usage Examples

```bash
# Basic usage with verbose output
TOOL_VERBOSE=1 LD_PRELOAD=./nixnan.so ./my_program

# Enable source line information (compile with -lineinfo)
LINE_INFO=1 LD_PRELOAD=./nixnan.so ./my_program

# Sample every 64th kernel invocation (for long-running programs)
SAMPLING=64 LD_PRELOAD=./nixnan.so ./my_program

# Log to file instead of stderr
LOGFILE=/tmp/nixnan.log LD_PRELOAD=./nixnan.so ./my_program

# Enable memory instrumentation
INSTR_MEM=1 LD_PRELOAD=./nixnan.so ./my_program

# Limit instrumentation to specific instruction range
INSTR_BEGIN=100 INSTR_END=500 LD_PRELOAD=./nixnan.so ./my_program

# Enable histogram tracking
HISTOGRAM=1 LD_PRELOAD=./nixnan.so ./my_program

# Combined: verbose, line info, and logging
TOOL_VERBOSE=1 LINE_INFO=1 LOGFILE=./debug.log LD_PRELOAD=./nixnan.so ./my_program
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

#### Targeted Range Monitoring

Create a JSON specification file:

```json
{
  "f32": {
    "ranges": [
      {"min": -126, "max": -120, "report_frequency": 1000},
      {"min": 120, "max": 127, "report_frequency": 100}
    ]
  },
  "f16": {
    "ranges": [
      {"min": -14, "max": -10, "report_frequency": 500}
    ]
  }
}
```

Run with specification:

```bash
BIN_SPEC_FILE=./ranges.json LD_PRELOAD=./nixnan.so ./my_program
```

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

### Case Study 3: CUDA GMRES Solver

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
