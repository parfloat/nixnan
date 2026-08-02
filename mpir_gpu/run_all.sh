#!/usr/bin/env bash
# Full Phase C protocol: build, calibrate, headline with 5 seeds, Phase D.
set -e
make
echo "== calibration sweep (TIMING=0) =="
TIMING=0 ./ir_gpu --n 4096 --allconfigs --sweep 1e1,3e1,1e2,3e2,1e3,3e3,1e4,3e4,1e5 --seed 0 | tee sweep.log
echo "== headline, 5 seeds x {C1,C2,C3}_gpu (edit kappas per sweep.log) =="
for s in 0 1 2 3 4; do
  TIMING=0 ./ir_gpu --n 4096 --allconfigs --sweep 1e2,1e3,3e4 --seed $s
done | tee headline.log
echo "== per-iteration detail at C2_gpu =="
TIMING=2 ./ir_gpu --n 4096 --allconfigs --kappa 1e3 --seed 0 | tee detail.log
echo "== Phase D: fp16 exceptions (overflow, then underflow stall) =="
TIMING=1 ./ir_gpu --n 4096 --config F16  --kappa 1e3 --noscale --bscale 1e5 --seed 0 | tee phased.log
TIMING=1 ./ir_gpu --n 4096 --config B16T --kappa 1e3 --noscale --bscale 1e5 --seed 0 | tee -a phased.log
TIMING=2 ./ir_gpu --n 4096 --config F16  --kappa 1e3 --noscale --seed 0 | tee -a phased.log
echo "== Phase E: Nix-Nan exception detection on overflow cases =="
LD_PRELOAD=../nixnan.so TIMING=1 ./ir_gpu --n 4096 --config F16  --kappa 1e3 --noscale --bscale 1e5 --seed 0 | tee nixnan.log
LD_PRELOAD=../nixnan.so TIMING=1 ./ir_gpu --n 4096 --config B16T --kappa 1e3 --noscale --bscale 1e5 --seed 0 | tee -a nixnan.log
LD_PRELOAD=../nixnan.so TIMING=1 ./ir_gpu --n 4096 --config F32  --kappa 1e3 --noscale --bscale 1e5 --seed 0 | tee -a nixnan.log
