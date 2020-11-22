include make.defs.hip

ifdef VERBOSE
    CPPFLAGS += -DVERBOSE
endif

SYCLFLAGS = $(SYCLFLAG)

.PHONY: all

all: stencil-hip stencil-sycl-hip stencil-2d-sycl-hip #stencil-sycl-usm-hip

stencil-sycl-hip: stencil-sycl.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

stencil-2d-sycl-hip: stencil-2d-sycl.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

stencil-sycl-usm-hip: stencil-sycl-usm.cc prk_util.h prk_sycl.h
	$(SYCLCXX) $(CPPFLAGS) $(SYCLFLAGS) $< -o $@

stencil-hip: stencil-hip.cc prk_util.h prk_hip.h
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $< -o $@

