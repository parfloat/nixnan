# Compiler and flags
NVCC = nvcc
NVCCFLAGS = -lineinfo -arch=sm_86

# Find all .cu source files in immediate subdirectories
SOURCES = $(wildcard */*.cu)

# Generate the corresponding target executable names by removing the .cu extension
TARGETS = $(patsubst %.cu,%,$(SOURCES))

# The default 'all' target depends on all the individual executables
.PHONY: all
all: $(TARGETS)

# Pattern rule to compile a .cu file into an executable.
# This rule matches any target that has a corresponding .cu file.
# Example: 'add/add_single' target will be built from 'add/add_single.cu'
%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Target to clean up all compiled executables
.PHONY: clean
clean:
	rm -f $(TARGETS)