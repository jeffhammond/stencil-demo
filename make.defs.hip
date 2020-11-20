CXXFLAGS=-g -O3 -std=c++17
#
# hipSYCL
#
SYCLDIR=/opt/hipSYCL
SYCLCXX=${SYCLDIR}/bin/syclcc-clang
SYCLFLAG=${CXXFLAGS}
SYCLFLAG+=-DHIPSYCL
SYCLFLAG+=--hipsycl-platform=rocm
SYCLFLAG+=--hipsycl-gpu-arch=gfx803
#
# HIP flags
#
HIPDIR=/opt/rocm-3.8.0
HIPCC=${HIPDIR}/bin/hipcc
HIPFLAGS=${CXXFLAGS}
