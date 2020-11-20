#
# This file shows the HIP+GCC toolchain options.
#
# Base compilers and language options
#
VERSION=
# C99 is required in some implementations.
CC=gcc${VERSION} -std=gnu11
#EXTRA_CLIBS=-lrt
# All of the Fortran code is written for the 2008 standard and requires preprocessing.
FC=gfortran${VERSION} -std=f2008 -cpp
# C++11 may not be required but does no harm here.
CXX=g++${VERSION} -std=gnu++17
#
# Compiler flags
#
# -mtune=native is appropriate for most cases.
# -march=native is appropriate if you want portable binaries.
DEFAULT_OPT_FLAGS=-O3 =g
DEFAULT_OPT_FLAGS+=-mtune=native -ffast-math
DEFAULT_OPT_FLAGS+=-mavx -mfma
DEFAULT_OPT_FLAGS+=-Wall #-Werror
#
# SYCL flags
#
# hipSYCL
#
SYCLDIR=/opt/hipSYCL
SYCLCXX=${SYCLDIR}/bin/syclcc-clang
SYCLFLAG=-std=c++17 -O3
SYCLFLAG+=-DHIPSYCL
SYCLFLAG+=--hipsycl-platform=rocm
SYCLFLAG+=--hipsycl-gpu-arch=gfx803
#
# HIP flags
#
HIPDIR=/opt/rocm-3.8.0
HIPCC=${HIPDIR}/bin/hipcc
HIPFLAGS=-g -O3 -std=c++11
