include make.defs.hip

ifdef VERBOSE
    CPPFLAGS += -DVERBOSE
endif

SYCLFLAGS = $(SYCLFLAG)

.PHONY: all

all: stencil-hip stencil-sycl stencil-2d-sycl #stencil-sycl-usm

%-sycl: %-sycl.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-sycl-usm: %-sycl-usm.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

%-hip: %-hip.cc prk_util.h prk_hip.h
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $< -o $@

