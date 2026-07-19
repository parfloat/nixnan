#!/bin/bash

# rd_nixnan_experiment.sh
# Experimental workflow for precision-edge CUDA demo with nixnan monitoring
# Tests floating-point precision limits across FP16, BF16, and FP32 formats
# using a reaction-diffusion equation (FTCS discretization)
#
# Demonstrates:
#  - Per-format exception detection (overflow, underflow, NaN propagation)
#  - Exponent histogram bucketing by format
#  - Early-warning tripwire detection before overflow
#  - Source-line attribution of inf/NaN events

set -e

# Configuration
NIXNAN_SO="${NIXNAN_PATH:-./nixnan.so}"
CUDA_ARCH="sm_80"  # A100 GPU; adjust to sm_86 if targeting Ampere H100
EXECUTABLE="./rd_nixnan"
SPEC_FILE="./spec.json"

echo "========================================"
echo "rd_nixnan Precision-Edge Demo Workflow"
echo "========================================"

# ============================================================================
# STAGE 1: Build the CUDA kernels (three formats: f16, bf16, f32)
# ============================================================================
echo ""
echo "[1/5] Building CUDA executable..."
echo "      Command: nvcc -arch=$CUDA_ARCH rd_nixnan.cu -o $EXECUTABLE"
nvcc -arch=$CUDA_ARCH rd_nixnan.cu -o $EXECUTABLE
if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: Build failed; $EXECUTABLE not found"
    exit 1
fi
echo "      ✓ Build successful"

# ============================================================================
# STAGE 2: Baseline run — no instrumentation
# ============================================================================
# Baseline shows native behavior: each format blows up at different step counts.
# FP16 overflows ~step 227; BF16/FP32 overflow ~step 1818.
# Watch for silent NaN propagation in max_element() and reported max of 0.0.
echo ""
echo "[2/5] Baseline run (no instrumentation)..."
echo "      Observing: step counts to overflow for each format (f16, bf16, f32)"
$EXECUTABLE
echo "      ✓ Baseline complete"

# ============================================================================
# STAGE 3: Exception summary per format (LD_PRELOAD nixnan.so)
# ============================================================================
# Counters: overflow (Inf), underflow (Subnormal), NaN propagation.
# Attribution per format, not per kernel, but shows which formats throw.
# FP16 unique: Subnormal count (diffusion term ~2⁻¹³·u dropping below 2⁻¹⁴).
echo ""
echo "[3/5] Exception summary with LD_PRELOAD..."
echo "      Counters per format: Inf, Subnormal, NaN"
echo "      Command: LD_PRELOAD=$NIXNAN_SO $EXECUTABLE"
LD_PRELOAD=$NIXNAN_SO $EXECUTABLE
echo "      ✓ Exception summary complete"

# ============================================================================
# STAGE 4: Exponent histogram bucketing
# ============================================================================
# HISTOGRAM=1 exposes the exponent distribution for each format.
# Teaching moment: FP16 max exponent 15 (2⁻¹⁴ < subnorm < 2⁻¹⁵),
#                   BF16/FP32 max exponent 127 (both ride full 8-bit exp).
# Contrast BF16 vs FP32 at same exponent: mantissa precision failure only shown by histogram width.
echo ""
echo "[4/5] Exponent histogram per format..."
echo "      Histogram reveals: FP16 [-14,15], BF16 [-5,127], FP32 [-5,127]"
echo "      Command: HISTOGRAM=1 LD_PRELOAD=$NIXNAN_SO $EXECUTABLE"
HISTOGRAM=1 LD_PRELOAD=$NIXNAN_SO $EXECUTABLE
echo "      ✓ Histogram complete"

# ============================================================================
# STAGE 5: Pre-overflow tripwire with spec.json
# ============================================================================
# spec.json defines bin thresholds (e.g., f16 [13,15], bf16/f32 [120,127]).
# Fires warnings as values enter high binades, giving early warning before inf.
# Nicer tutorial arc: anomalies detected _before_ the overflow corpse appears.
echo ""
echo "[5/5] Pre-overflow tripwire detection..."
echo "      Warnings triggered as values approach format max exponent"
echo "      Command: BIN_SPEC_FILE=$SPEC_FILE HISTOGRAM=1 LD_PRELOAD=$NIXNAN_SO $EXECUTABLE"
BIN_SPEC_FILE=$SPEC_FILE HISTOGRAM=1 LD_PRELOAD=$NIXNAN_SO $EXECUTABLE
echo "      ✓ Tripwire detection complete"

# ============================================================================
# OPTIONAL: Full instrumentation with source-line attribution
# ============================================================================
# Uncomment to enable SASS instruction memory and source-line tracking.
# This catches the exact stencil operation (inf - 2·inf + inf → NaN) and
# reports which source line in the kernel triggered it.
# Overhead: 50× NVBit multiplier, but ~7.4e5 step-ops is still sub-second.
#
# echo ""
# echo "[OPTIONAL] Full instrumentation with source-line attribution..."
# echo "      Command: INSTR_MEM=1 LINE_INFO=1 HISTOGRAM=1 LD_PRELOAD=$NIXNAN_SO $EXECUTABLE"
# INSTR_MEM=1 LINE_INFO=1 HISTOGRAM=1 LD_PRELOAD=$NIXNAN_SO $EXECUTABLE
# echo "      ✓ Full instrumentation complete"

echo ""
echo "========================================"
echo "Workflow complete."
echo "========================================"
echo ""
echo "Interpretation guide:"
echo "  • FP16: Overflow + Subnormal + NaN (diffusion flushed, reaction runs away)"
echo "  • BF16: Overflow + NaN (diffusion noise throughout, precision loss not flagged)"
echo "  • FP32: Overflow + NaN (clean mantissa, exponent range is teaching contrast vs BF16)"
echo ""
echo "If FP16 subnormal count is zero, adjust demo parameters:"
echo "  - Lower lambda (line 13 of tutorial) or increase dt to push boundary diffusion below 2⁻¹⁴"
echo ""
