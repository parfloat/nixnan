# Builds both precision variants of reaction_diffusion_gpu.cu, mirroring
# the CPU tutorial's "typedef double Real_t; // typedef float Real_t;"
# toggle as a compile-time macro instead of hand-editing the source.
#
# -lineinfo is required for nixnan's LINE_INFO=1 source-location reporting.

NVCC = nvcc
ARCH = sm_86
NVCCFLAGS = -std=c++17 -g -O2 -lineinfo -arch=$(ARCH)

SRC = reaction_diffusion_gpu.cu
FP64_EXE = reaction_diffusion_gpu_fp64
FP32_EXE = reaction_diffusion_gpu_fp32

all: $(FP64_EXE) $(FP32_EXE)

$(FP64_EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(FP32_EXE): $(SRC)
	$(NVCC) $(NVCCFLAGS) -DRD_USE_FLOAT -o $@ $<

clean:
	rm -f $(FP64_EXE) $(FP32_EXE)

.PHONY: all clean
