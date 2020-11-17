include make.defs

# debugging
ifdef VERBOSE
    CPPFLAGS += -DVERBOSE
endif

CXXFLAGS  = $(DEFAULT_OPT_FLAGS) $(CPPFLAGS)

#BOOSTFLAGS = $(BOOSTFLAG)
#RANGEFLAGS = $(RANGEFLAG) -DUSE_RANGES
SYCLFLAGS = $(SYCLFLAG)

ifdef OCCADIR
  include ${OCCADIR}/scripts/makefile
endif
OCCAFLAGS = -I${OCCADIR}/include -Wl,-rpath -Wl,${OCCADIR}/lib -L${OCCADIR}/lib -locca

.PHONY: all clean cuda sycl

all: stencil-cuda stencil-sycl stencil-2d-sycl stencil-sycl-usm

%-sycl: %-sycl.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-sycl-explicit-usm: %-sycl-explicit-usm.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-sycl-usm: %-sycl-usm.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-sycl-explicit: %-sycl-explicit.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-cuda: %-cuda.cu prk_util.h prk_cuda.h
	$(NVCC) $(CUDAFLAGS) $(CPPFLAGS) $< -o $@

%: %.cc prk_util.h
	$(CXX) $(CXXFLAGS) $< -o $@

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
	-rm -f *-opencl

cleancl:
	-rm -f star[123456789].cl
	-rm -f grid[123456789].cl
