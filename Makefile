# .PHONY: NVBIT GPU-FPX 

all:analyzer detector nixnan nixnan.so

nvbit_version = 1.7.5

nvbit_tar=nvbit-Linux-x86_64-$(nvbit_version).tar.bz2
nvbit_tool=$(shell pwd)/nvbit_release/tools
GPUFPX_home=$(nvbit_tool)/GPU-FPX

analyzer: $(GPUFPX_home)/analyzer/analyzer.so
detector: $(GPUFPX_home)/detector/detector.so
nixnan: $(nvbit_tool)/nixnan/nixnan.so

$(GPUFPX_home)/analyzer/analyzer.so: $(GPUFPX_home)/analyzer/analyzer.cu $(nvbit_tar)
	cd $(GPUFPX_home)/analyzer; \
	$(MAKE)

$(GPUFPX_home)/detector/detector.so: $(GPUFPX_home)/detector/detector.cu $(nvbit_tar)
	cd $(GPUFPX_home)/detector; \
	$(MAKE)
nixnan.so: $(nvbit_tool)/nixnan/nixnan.so
	ln -sf $(nvbit_tool)/nixnan/nixnan.so nixnan.so

$(nvbit_tool)/nixnan/nixnan.so: $(nvbit_tool)/nixnan/nixnan.cu $(nvbit_tar)
	cd $(nvbit_tool)/nixnan; \
	$(MAKE)

$(nvbit_tar):
	wget https://github.com/NVlabs/NVBit/releases/download/v$(nvbit_version)/$@
	tar -xf $(nvbit_tar)
	cp -R nvbit_release_x86_64/* nvbit_release
clean:
	rm nvbit-Linux-x86_64-$(nvbit_version).tar.bz2