# Mixed-precision Cholesky-IR on A100 -- Phase C/D of IR.md
NVCC   ?= nvcc
ARCH   ?= sm_80
CFLAGS  = -O3 -arch=$(ARCH) -std=c++14 --expt-relaxed-constexpr
LIBS    = -lcublas -lcusolver -lcurand

all: ir_gpu

ir_gpu: Code2.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LIBS)

# Headline: 5 configs x {C1,C2,C3}_gpu, genuine loop timing (TIMING=0)
run: ir_gpu
	TIMING=0 ./ir_gpu --n 4096 --allconfigs --sweep 1e2,1e3,3e4 --seed 0

# Calibration sweep on hardware (thresholds shift with n vs Colab)
sweep: ir_gpu
	TIMING=0 ./ir_gpu --n 4096 --allconfigs \
	  --sweep 1e1,3e1,1e2,3e2,1e3,3e3,1e4,3e4,1e5 --seed 0

# Per-iteration timing detail at C2_gpu
detail: ir_gpu
	TIMING=2 ./ir_gpu --n 4096 --allconfigs --kappa 1e3 --seed 0

# Phase D: force FP16 exceptions.
# d1: overflow -- unscaled correction path + amplified b: residual rounds to
#     fp16 Inf immediately (BF16 run for contrast stays clean).
# d2: underflow stall -- unscaled path only: FP16 residual flushes subnormal.
phase_d: ir_gpu
	TIMING=1 ./ir_gpu --n 4096 --config F16  --kappa 1e3 --noscale --bscale 1e5 --seed 0
	TIMING=1 ./ir_gpu --n 4096 --config B16T --kappa 1e3 --noscale --bscale 1e5 --seed 0
	TIMING=2 ./ir_gpu --n 4096 --config F16  --kappa 1e3 --noscale --seed 0

# Phase E: Nix-Nan exception detection on overflow cases
nixnan: ir_gpu
	LD_PRELOAD=../nixnan.so TIMING=1 ./ir_gpu --n 4096 --config F16  --kappa 1e3 --noscale --bscale 1e5 --seed 0
	LD_PRELOAD=../nixnan.so TIMING=1 ./ir_gpu --n 4096 --config B16T --kappa 1e3 --noscale --bscale 1e5 --seed 0
	LD_PRELOAD=../nixnan.so TIMING=1 ./ir_gpu --n 4096 --config F32  --kappa 1e3 --noscale --bscale 1e5 --seed 0

clean:
	rm -f ir_gpu
.PHONY: all run sweep detail phase_d nixnan clean
