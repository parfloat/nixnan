#!/usr/bin/env bash
# Safe refinement run: well-conditioned matrices (n=256, κ≤1000) with convergence data
# Includes Nix-Nan exception detection on all runs
set -e

make

echo "== Safe convergence sweep (n=256, κ∈{10,30,100,300,1000}) =="
TIMING=2 ./ir_gpu --n 256 --allconfigs --kappa 10 --seed 0 | tee safe_refine.log
TIMING=2 ./ir_gpu --n 256 --allconfigs --kappa 30 --seed 0 | tee -a safe_refine.log
TIMING=2 ./ir_gpu --n 256 --allconfigs --kappa 100 --seed 0 | tee -a safe_refine.log
TIMING=2 ./ir_gpu --n 256 --allconfigs --kappa 300 --seed 0 | tee -a safe_refine.log
TIMING=2 ./ir_gpu --n 256 --allconfigs --kappa 1000 --seed 0 | tee -a safe_refine.log

echo "== Safe convergence with Nix-Nan instrumentation =="
LD_PRELOAD=../nixnan.so TIMING=2 ./ir_gpu --n 256 --allconfigs --kappa 100 --seed 0 | tee safe_nixnan.log
LD_PRELOAD=../nixnan.so TIMING=2 ./ir_gpu --n 256 --allconfigs --kappa 1000 --seed 0 | tee -a safe_nixnan.log
