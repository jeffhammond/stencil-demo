#include "prk_util.h"
#include "prk_cuda.h"

__global__ void star2(const int n, const double * in, double * out) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ( (2 <= i) && (i < n-2) && (2 <= j) && (j < n-2) ) {
            out[i*n+j] += +in[(i)*n+(j-2)] * -0.125
                          +in[(i)*n+(j-1)] * -0.25
                          +in[(i-2)*n+(j)] * -0.125
                          +in[(i-1)*n+(j)] * -0.25
                          +in[(i+1)*n+(j)] * 0.25
                          +in[(i+2)*n+(j)] * 0.125
                          +in[(i)*n+(j+1)] * 0.25
                          +in[(i)*n+(j+2)] * 0.125;
     }
}

__global__ void star3(const int n, const double * in, double * out) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ( (3 <= i) && (i < n-3) && (3 <= j) && (j < n-3) ) {
            out[i*n+j] += +in[(i)*n+(j-3)] * -0.05555555555555555
                          +in[(i)*n+(j-2)] * -0.08333333333333333
                          +in[(i)*n+(j-1)] * -0.16666666666666666
                          +in[(i-3)*n+(j)] * -0.05555555555555555
                          +in[(i-2)*n+(j)] * -0.08333333333333333
                          +in[(i-1)*n+(j)] * -0.16666666666666666
                          +in[(i+1)*n+(j)] * 0.16666666666666666
                          +in[(i+2)*n+(j)] * 0.08333333333333333
                          +in[(i+3)*n+(j)] * 0.05555555555555555
                          +in[(i)*n+(j+1)] * 0.16666666666666666
                          +in[(i)*n+(j+2)] * 0.08333333333333333
                          +in[(i)*n+(j+3)] * 0.05555555555555555;
     }
}

__global__ void star4(const int n, const double * in, double * out) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ( (4 <= i) && (i < n-4) && (4 <= j) && (j < n-4) ) {
            out[i*n+j] += +in[(i)*n+(j-4)] * -0.03125
                          +in[(i)*n+(j-3)] * -0.041666666666666664
                          +in[(i)*n+(j-2)] * -0.0625
                          +in[(i)*n+(j-1)] * -0.125
                          +in[(i-4)*n+(j)] * -0.03125
                          +in[(i-3)*n+(j)] * -0.041666666666666664
                          +in[(i-2)*n+(j)] * -0.0625
                          +in[(i-1)*n+(j)] * -0.125
                          +in[(i+1)*n+(j)] * 0.125
                          +in[(i+2)*n+(j)] * 0.0625
                          +in[(i+3)*n+(j)] * 0.041666666666666664
                          +in[(i+4)*n+(j)] * 0.03125
                          +in[(i)*n+(j+1)] * 0.125
                          +in[(i)*n+(j+2)] * 0.0625
                          +in[(i)*n+(j+3)] * 0.041666666666666664
                          +in[(i)*n+(j+4)] * 0.03125;
     }
}

__global__ void nothing(const int n, const double * in, double * out)
{
}

__global__ void add(const int n, double * in)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i<n) && (j<n)) {
        in[i*n+j] += 1.0;
    }
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << std::endl;
  std::cout << "C++11/CUDA Stencil execution on 2D grid" << std::endl;

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations;
  size_t n, block_size = 16, radius = 2;

  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <array dimension> [<block size> <stencil radius>]";
      }

      // number of times to run the algorithm
      iterations  = std::atoi(argv[1]);
      if (iterations < 1) {
        throw "ERROR: iterations must be >= 1";
      }

      // linear grid dimension
      n  = std::atoi(argv[2]);
      if (n < 1) {
        throw "ERROR: grid dimension must be positive";
      } else if (n > prk::get_max_matrix_size()) {
        throw "ERROR: grid dimension too large - overflow risk";
      }

      if (argc > 3) {
          block_size = std::atoi(argv[3]);
          if (block_size <= 0) block_size = n;
          if (block_size > n) block_size = n;
      }
      if (n % block_size) {
        throw "ERROR: block size does not evenly divide grid size";
      }

      // stencil radius
      radius = 2;
      if (argc > 4) {
          radius = std::atoi(argv[4]);
      }

      if ( (radius < 1) || (2*radius+1 > n) ) {
        throw "ERROR: Stencil radius negative or too large";
      }
  }
  catch (const char * e) {
    std::cout << e << std::endl;
    return 1;
  }

  std::cout << "Number of iterations = " << iterations << std::endl;
  std::cout << "Grid size            = " << n << std::endl;
  std::cout << "Block size           = " << block_size << std::endl;
  std::cout << "Radius of stencil    = " << radius << std::endl;

  //////////////////////////////////////////////////////////////////////
  /// Setup CUDA environment
  //////////////////////////////////////////////////////////////////////

  prk::CUDA::info info;
  info.print(1);

  auto stencil = nothing;
  switch (radius) {
      case 2: stencil = star2; break;
      case 3: stencil = star3; break;
      case 4: stencil = star4; break;
  }

  dim3 dimGrid(prk::divceil(n,block_size),prk::divceil(n,block_size),1);
  dim3 dimBlock(block_size, block_size, 1);
  info.checkDims(dimBlock, dimGrid);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double stencil_time{0};

  const size_t nelems = n*n;
  const size_t bytes = nelems * sizeof(double);
  double * h_in;
  double * h_out;
  prk::CUDA::check( cudaMallocHost((void**)&h_in, bytes) );
  prk::CUDA::check( cudaMallocHost((void**)&h_out, bytes) );

  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      h_in[i*n+j]  = static_cast<double>(i+j);
      h_out[i*n+j] = static_cast<double>(0);
    }
  }

  // copy input from host to device
  double * d_in;
  double * d_out;
  prk::CUDA::check( cudaMalloc((void**)&d_in, bytes) );
  prk::CUDA::check( cudaMalloc((void**)&d_out, bytes) );
  prk::CUDA::check( cudaMemcpy(d_in, &(h_in[0]), bytes, cudaMemcpyHostToDevice) );
  prk::CUDA::check( cudaMemcpy(d_out, &(h_out[0]), bytes, cudaMemcpyHostToDevice) );
  prk::CUDA::check( cudaDeviceSynchronize() );

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) stencil_time = prk::wtime();

    // Apply the stencil operator
    stencil<<<dimGrid, dimBlock>>>(n, d_in, d_out);

    // Add constant to solution to force refresh of neighbor data, if any
    add<<<dimGrid, dimBlock>>>(n, d_in);

    prk::CUDA::check( cudaDeviceSynchronize() );
  }
  stencil_time = prk::wtime() - stencil_time;

  // copy output back to host
  prk::CUDA::check( cudaMemcpy(&(h_out[0]), d_out, bytes, cudaMemcpyDeviceToHost) );

#ifdef VERBOSE
  // copy input back to host - debug only
  prk::CUDA::check( cudaMemcpy(&(h_in[0]), d_in, bytes, cudaMemcpyDeviceToHost) );
#endif

  prk::CUDA::check( cudaFree(d_out) );
  prk::CUDA::check( cudaFree(d_in) );

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  // interior of grid with respect to stencil
  const size_t active_points = (n-2L*radius)*(n-2L*radius);
  double norm{0};
  for (size_t i=radius; i<n-radius; i++) {
    for (size_t j=radius; j<n-radius; j++) {
      norm += prk::abs(h_out[i*n+j]);
    }
  }
  norm /= active_points;

  // verify correctness
  const double epsilon = 1.0e-8;
  const double reference_norm = 2*(iterations+1);
  if (prk::abs(norm-reference_norm) > epsilon) {
    std::cout << "ERROR: L1 norm = " << norm
              << " Reference L1 norm = " << reference_norm << std::endl;
    return 1;
  } else {
    std::cout << "Solution validates" << std::endl;
#ifdef VERBOSE
    std::cout << "L1 norm = " << norm
              << " Reference L1 norm = " << reference_norm << std::endl;
#endif
    const size_t stencil_size = 4*radius+1;
    size_t flops = (2L*stencil_size+1L) * active_points;
    double avgtime = stencil_time/iterations;
    std::cout << 8*sizeof(double) << "B "
              << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}
