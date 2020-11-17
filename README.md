# Getting Started

## DPC++

Git hash: d81081f70085de828d2ff8acdd3e62621af5d00c





# Results

## CUDA

```
jhammond@thetagpu13:~/PRK/Cxx11/sandbox$ for t in 1 2 4 8 16 32 ; do ./stencil-cuda 100 8000 $t star 4 ; done
Parallel Research Kernels version 
C++11/CUDA Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Tile size            = 1
Type of stencil      = star
Radius of stencil    = 4
Solution validates
Rate (MFlops/s): 23745.6 Avg time (s): 0.0941447
Parallel Research Kernels version 
C++11/CUDA Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Tile size            = 2
Type of stencil      = star
Radius of stencil    = 4
Solution validates
Rate (MFlops/s): 94231.9 Avg time (s): 0.0237236
Parallel Research Kernels version 
C++11/CUDA Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Tile size            = 4
Type of stencil      = star
Radius of stencil    = 4
Solution validates
Rate (MFlops/s): 347783 Avg time (s): 0.00642793
Parallel Research Kernels version 
C++11/CUDA Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Tile size            = 8
Type of stencil      = star
Radius of stencil    = 4
Solution validates
Rate (MFlops/s): 425630 Avg time (s): 0.00525227
Parallel Research Kernels version 
C++11/CUDA Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Tile size            = 16
Type of stencil      = star
Radius of stencil    = 4
Solution validates
Rate (MFlops/s): 245849 Avg time (s): 0.00909305
Parallel Research Kernels version 
C++11/CUDA Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Tile size            = 32
Type of stencil      = star
Radius of stencil    = 4
Solution validates
Rate (MFlops/s): 223730 Avg time (s): 0.00999204
```

## SYCL

```
jhammond@thetagpu13:~/PRK/Cxx11/sandbox$ for t in 1 2 4 8 16 32 ; do ./stencil-sycl 100 8000 $t star 4 ; done
Parallel Research Kernels version 
C++11/SYCL Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Block size           = 1
Type of stencil      = star
Radius of stencil    = 4
SYCL Device:   A100-SXM4-40GB
SYCL Platform: NVIDIA CUDA BACKEND
Solution validates
64B Rate (MFlops/s): 46322.8 Avg time (s): 0.0482597
Parallel Research Kernels version 
C++11/SYCL Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Block size           = 2
Type of stencil      = star
Radius of stencil    = 4
SYCL Device:   A100-SXM4-40GB
SYCL Platform: NVIDIA CUDA BACKEND
Solution validates
64B Rate (MFlops/s): 165960 Avg time (s): 0.0134702
Parallel Research Kernels version 
C++11/SYCL Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Block size           = 4
Type of stencil      = star
Radius of stencil    = 4
SYCL Device:   A100-SXM4-40GB
SYCL Platform: NVIDIA CUDA BACKEND
Solution validates
64B Rate (MFlops/s): 486228 Avg time (s): 0.00459768
Parallel Research Kernels version 
C++11/SYCL Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Block size           = 8
Type of stencil      = star
Radius of stencil    = 4
SYCL Device:   A100-SXM4-40GB
SYCL Platform: NVIDIA CUDA BACKEND
Solution validates
64B Rate (MFlops/s): 1.00311e+06 Avg time (s): 0.0022286
Parallel Research Kernels version 
C++11/SYCL Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Block size           = 16
Type of stencil      = star
Radius of stencil    = 4
SYCL Device:   A100-SXM4-40GB
SYCL Platform: NVIDIA CUDA BACKEND
Solution validates
64B Rate (MFlops/s): 1.0703e+06 Avg time (s): 0.00208868
Parallel Research Kernels version 
C++11/SYCL Stencil execution on 2D grid
Number of iterations = 100
Grid size            = 8000
Block size           = 32
Type of stencil      = star
Radius of stencil    = 4
SYCL Device:   A100-SXM4-40GB
SYCL Platform: NVIDIA CUDA BACKEND
Solution validates
64B Rate (MFlops/s): 1.06991e+06 Avg time (s): 0.00208946
```
