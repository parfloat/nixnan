#!/usr/bin/env bash
# Comprehensive Nix-Nan instrumentation for κ=10 and κ=1000 with all precisions
set -e

echo "=== Nix-Nan for κ=10 (low condition number, good convergence) ==="
for prec in F64 F32 TF32 F16 B16T B16S; do
  echo "  Running $prec..."
  LD_PRELOAD=../nixnan.so TIMING=2 ./ir_gpu --n 256 --config $prec --kappa 10 --seed 0 | tee nixnan_log_10_${prec}.log
done

echo ""
echo "=== Nix-Nan for κ=1000 (high condition number, challenging convergence) ==="
for prec in F64 F32 TF32 F16 B16T B16S; do
  echo "  Running $prec..."
  LD_PRELOAD=../nixnan.so TIMING=2 ./ir_gpu --n 256 --config $prec --kappa 1000 --seed 0 | tee nixnan_log_1000_${prec}.log
done

echo ""
echo "=== Nix-Nan runs completed ==="
