#!/usr/bin/env bash
# Phase 8: Binary search for exception threshold in well-conditioned matrix regime
# Start from κ=1000 (no exceptions) and step up until we find exception-generating κ
set -e

mkdir -p phase8_logs

echo "=== Phase 8: Finding Exception Threshold ==="
echo "Starting from κ=1000 (known safe) and stepping up to find first exception..."
echo ""

# Test condition numbers in increasing order
# Start with κ=1000 (known safe) and step up
test_kappas=(1000 2000 3000 5000 10000 20000 30000)

for k in "${test_kappas[@]}"; do
  echo "Testing κ=$k..."

  # Run all 6 formats with Nix-Nan at this κ
  for prec in F64 F32 TF32 F16 B16T B16S; do
    logfile="phase8_logs/phase8_κ${k}_${prec}.log"
    echo "  $prec → $logfile"

    LD_PRELOAD=../nixnan.so TIMING=2 ./ir_gpu --n 256 --config $prec --kappa $k --seed 0 2>&1 | tee "$logfile"

    # Check for exceptions
    if grep -q "#nixnan: error" "$logfile"; then
      echo "    ✓ EXCEPTION FOUND in $prec at κ=$k"
      grep "#nixnan: error" "$logfile" | head -5 >> "phase8_logs/EXCEPTIONS_FOUND.txt"
    else
      echo "    - No exceptions detected"
    fi
  done

  echo ""
done

echo "=== Phase 8 Complete ==="
echo "Results saved in phase8_logs/"
ls -lh phase8_logs/
