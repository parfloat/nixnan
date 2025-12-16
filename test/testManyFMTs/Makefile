# Makefile for testManyFMTs

NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_75 -lcurand

# Detect GPU architecture if possible
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifneq ($(GPU_ARCH),)
    NVCC_FLAGS = -O2 -arch=sm_$(GPU_ARCH) -lcurand
endif

TARGET = testManyFMTs
SRC = testManyFMTs.cu

.PHONY: all clean run run-nixnan

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Run with nixnan instrumentation
NIXNAN_SO ?= ../../nixnan.so
run-nixnan: $(TARGET)
	LD_PRELOAD=$(NIXNAN_SO) ./$(TARGET)
