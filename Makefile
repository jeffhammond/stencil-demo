.PHONY: all clean

all:
	@echo "To build programs, run 'make -f Makefile.x -j -k' for x={oneapi,cuda,hip}"

clean:
	-rm -f *.o
	-rm -f *.s
	-rm -f *.ll # Coriander
	-rm -f pgipar* # PGI?
	-rm -f *.optrpt
	-rm -f *.dwarf
	-rm -rf *.dSYM # Mac
	-rm -f *-sycl
	-rm -f *-sycl-usm
	-rm -f *-sycl-explicit
	-rm -f *-sycl-explicit-usm
	-rm -f *-cuda
	-rm -f *-hip
	-rm -f *-oneapi
	-rm -f *-opencl
