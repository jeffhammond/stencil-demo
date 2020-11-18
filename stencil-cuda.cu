#include "prk_util.h"
#include "prk_cuda.h"

__global__ void star2(const int n, const double * in, double * out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
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
        in[i*n+j] += (double)1;
    }
}

int main(int argc, char* argv[])
{
  std::cout << "Parallel Research Kernels version " << std::endl;
  std::cout << "C++11/CUDA Stencil execution on 2D grid" << std::endl;

  prk::CUDA::info info;
  //info.print();

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  int iterations, n, radius, tile_size;
  bool star = true;
  try {
      if (argc < 3) {
        throw "Usage: <# iterations> <array dimension> [<tile_size> <star/grid> <radius>]";
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

      // default tile size for tiling of local transpose
      tile_size = 32;
      if (argc > 3) {
          tile_size = std::atoi(argv[3]);
          if (tile_size <= 0) tile_size = n;
          if (tile_size > n) tile_size = n;
          if (tile_size > 32) {
              std::cout << "Warning: tile_size > 32 may lead to incorrect results (observed for CUDA 9.0 on GV100).\n";
          }
      }

      // stencil pattern
      if (argc > 4) {
          auto stencil = std::string(argv[4]);
          auto grid = std::string("grid");
          star = (stencil == grid) ? false : true;
      }

      // stencil radius
      radius = 2;
      if (argc > 5) {
          radius = std::atoi(argv[5]);
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
  std::cout << "Tile size            = " << tile_size << std::endl;
  std::cout << "Type of stencil      = " << (star ? "star" : "grid") << std::endl;
  std::cout << "Radius of stencil    = " << radius << std::endl;

  auto stencil = nothing;
  if (star) {
      switch (radius) {
          case 2: stencil = star2; break;
          case 3: stencil = star3; break;
          case 4: stencil = star4; break;
      }
  }

  dim3 dimGrid(prk::divceil(n,tile_size),prk::divceil(n,tile_size),1);
  dim3 dimBlock(tile_size, tile_size, 1);
  info.checkDims(dimBlock, dimGrid);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double stencil_time{0};

  const size_t nelems = (size_t)n * (size_t)n;
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
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // interior of grid with respect to stencil
  size_t active_points = static_cast<size_t>(n-2*radius)*static_cast<size_t>(n-2*radius);
  double norm = 0.0;
  for (int i=radius; i<n-radius; i++) {
    for (int j=radius; j<n-radius; j++) {
      norm += prk::abs(h_out[i*n+j]);
    }
  }
  norm /= active_points;

  // verify correctness
  const double epsilon = 1.0e-8;
  double reference_norm = 2.*(iterations+1.);
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
    const int stencil_size = star ? 4*radius+1 : (2*radius+1)*(2*radius+1);
    size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
    auto avgtime = stencil_time/iterations;
    std::cout << "Rate (MFlops/s): " << 1.0e-6 * static_cast<double>(flops)/avgtime
              << " Avg time (s): " << avgtime << std::endl;
  }

  return 0;
}
