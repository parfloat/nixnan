# nixnan regression tests.   `make help` lists the targets.

NVCC      ?= nvcc
ARCH      ?= native
NVCCFLAGS ?= -lineinfo -arch=$(ARCH)
PY        ?= python3
NIXNAN_SO ?= $(abspath ../nixnan.so)

# Every test is a */*.cu file. RUNNABLE is the subset that applies to the GPU in this machine
# (decided by run.py from the // @min_sm / @max_sm annotations), so `make` never tries to build
# a BF16 or TF32 test on a GPU that cannot run it.
SOURCES  := $(wildcard */*.cu)
TARGETS  := $(patsubst %.cu,%,$(SOURCES))
RUNNABLE := $(shell $(PY) run.py --list-runnable 2>/dev/null)

.PHONY: all help list platform check record record-platform coverage gen clean

all: $(RUNNABLE)

help:
	@echo "make                     build every test that applies to this GPU"
	@echo "make <dir>/<test>        build one test, e.g.  make fp32/div_single"
	@echo "make run/<dir>/<test>    build + run one test under nixnan + compare, e.g.  make run/fp32/div_single"
	@echo "make run/<dir>           same for a whole directory, e.g.  make run/fp16"
	@echo "make check               build + run + compare everything (exit 1 on any failure)"
	@echo "make record              (re)write <test>.expect from the current output"
	@echo "make record-platform     (re)write <test>.sm<NN>.expect for this GPU"
	@echo "make coverage            which instrumented opcodes do the tests exercise?"
	@echo "make gen                 regenerate mma/wmma_*.cu from mma/gen_wmma_tests.py"
	@echo "make platform | list     show detected platform | runnable tests"
	@echo "make clean               remove built executables"
	@echo "Variables: NIXNAN_SO=$(NIXNAN_SO)  ARCH=$(ARCH)  NIXNAN_TEST_SM=<sm> (pretend GPU, for testing skips)"

platform:
	@$(PY) run.py --platform

list:
	@$(PY) run.py --list-runnable

# Build one test:  make fp32/div_single
%: %.cu helpers.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $<

define need_so
	@test -f $(NIXNAN_SO) || { echo "nixnan.so not found at $(NIXNAN_SO); build it first (make -C ..)"; exit 1; }
endef

# Build + run + compare one test or a directory/prefix:  make run/fp32/div_single   make run/fp16
run/%:
	$(need_so)
	@set -e; t="$(filter $*%,$(RUNNABLE))"; [ -z "$$t" ] || $(MAKE) --no-print-directory $$t
	$(PY) run.py $(NIXNAN_SO) --only $*

check: all
	$(need_so)
	$(PY) run.py $(NIXNAN_SO)

record: all
	$(need_so)
	$(PY) run.py $(NIXNAN_SO) --record

record-platform: all
	$(need_so)
	$(PY) run.py $(NIXNAN_SO) --record-platform

coverage: all
	$(PY) coverage.py

gen:
	cd mma && $(PY) gen_wmma_tests.py

clean:
	rm -f $(TARGETS)
