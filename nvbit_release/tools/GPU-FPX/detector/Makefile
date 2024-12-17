mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=$(current_dir).so

include ../utility/config.mk
SUBMAKEFILES = $(shell find . -maxdepth 4 -type f -name Makefile) 
SUBDIRS   = $(filter-out ./,$(dir $(SUBMAKEFILES)))
.PHONY : $(SUBDIRS)



NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

NVBIT_PATH=../../../core
UTILITY=../
INCLUDES=-I$(NVBIT_PATH) -I$(UTILITY)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)

OBJECTS=$(SOURCES:.cu=.o)
#ARCH?=35
#ARCH?=70

# mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
# current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
#
# NVBIT_TOOL=$(current_dir).so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=sm_$(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

%.o: %.cu 
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=sm_$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=sm_$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

test: SUBDIRS_TEST

SUBDIRS_TEST: $(SUBDIRS)
	export TOOL_VERBOSE=1
	export ENABLE_FUN_DETAIL=1
	export PRINT_ILL_INSTR=1
	for dir in $(SUBDIRS); do \
		(cd $$dir && make && python3 ../../../verify_except.py ./main && cd -;) \
	done

clean:
	rm -f *.so *.o

clean_test: 
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
