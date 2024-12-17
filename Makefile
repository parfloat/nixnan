# .PHONY: NVBIT GPU-FPX 

all:analyzer detector

nvbit_version = 1.7.2

nvbit_tar=nvbit-Linux-x86_64-$(nvbit_version).tar.bz2
nvbit_tool=$(shell pwd)/nvbit_release/tools
GPUFPX_home=$(nvbit_tool)/GPU-FPX

analyzer: $(GPUFPX_home)/analyzer/analyzer.so
detector: $(GPUFPX_home)/detector/detector.so

$(GPUFPX_home)/analyzer/analyzer.so: $(GPUFPX_home)/analyzer/analyzer.cu
	cd $(GPUFPX_home)/analyzer; \
	$(MAKE)

$(GPUFPX_home)/detector/detector.so: $(GPUFPX_home)/detector/detector.cu
	cd $(GPUFPX_home)/detector; \
	$(MAKE)

nvbit_release: $(nvbit_tar)
	tar -xf $<

$(nvbit_tar):
	wget https://github.com/NVlabs/NVBit/releases/download/v$(nvbit_version)/$@

clean:
	rm -rf nvbit_release/
	rm nvbit-Linux-x86_64-$(nvbit_version).tar.bz2