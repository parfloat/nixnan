# Build lu_solve_gpu for use as a nixnan instrumentation target.
#
# -lineinfo is required for nixnan's LINE_INFO=1 source-location reporting
# (see ../../Tutorial.md, "Basic Usage").

NVCC = nvcc
ARCH = sm_86
NVCCFLAGS = -std=c++17 -g -O2 -lineinfo -arch=$(ARCH)

SRC = lu_solve_gpu.cu
EXECUTABLE_NAME = lu_solve_gpu

all: $(EXECUTABLE_NAME)

$(EXECUTABLE_NAME): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(EXECUTABLE_NAME)

.PHONY: all clean
