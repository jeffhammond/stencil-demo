#include "prk_util.h"
#include "prk_sycl.h"

void add(sycl::queue & q, const size_t n, sycl::buffer<double> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2> {n,n}, [=] (sycl::id<2> it) {
      const size_t i = it.get(0);
      const size_t j = it.get(1);
      out[i*n+j] += static_cast<double>(1);
    });
  });
}

void add(sycl::queue & q, const size_t n, const size_t block_size, sycl::buffer<double> & d_out)
{
  sycl::range<2> global{n,n};
  sycl::range<2> local{block_size,block_size};
  q.submit([&](sycl::handler& h) {
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
      const size_t i = it.get_global_id(0);
      const size_t j = it.get_global_id(1);
      out[i*n+j] += static_cast<double>(1);
    });
  });
}

void star2(sycl::queue & q, const size_t n, sycl::buffer<double> & d_in, sycl::buffer<double> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2> {n,n}, [=] (sycl::id<2> it) {
      const size_t i = it.get(0);
      const size_t j = it.get(1);
      if ( (2 <= i) && (i < n-2) && (2 <= j) && (j < n-2) ) {
        out[i*n+j] += +in[i*n+(j+1)] * static_cast<double>(0.25)
                      +in[i*n+(j-1)] * static_cast<double>(-0.25)
                      +in[(i+1)*n+j] * static_cast<double>(0.25)
                      +in[(i-1)*n+j] * static_cast<double>(-0.25)
                      +in[i*n+(j+2)] * static_cast<double>(0.125)
                      +in[i*n+(j-2)] * static_cast<double>(-0.125)
                      +in[(i+2)*n+j] * static_cast<double>(0.125)
                      +in[(i-2)*n+j] * static_cast<double>(-0.125);
      }
    });
  });
}

void star3(sycl::queue & q, const size_t n, sycl::buffer<double> & d_in, sycl::buffer<double> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2> {n,n}, [=] (sycl::id<2> it) {
      const size_t i = it.get(0);
      const size_t j = it.get(1);
      if ( (3 <= i) && (i < n-3) && (3 <= j) && (j < n-3) ) {
        out[i*n+j] += +in[i*n+(j+1)] * static_cast<double>(0.166666666667)
                              +in[i*n+(j-1)] * static_cast<double>(-0.166666666667)
                              +in[(i+1)*n+j] * static_cast<double>(0.166666666667)
                              +in[(i-1)*n+j] * static_cast<double>(-0.166666666667)
                              +in[i*n+(j+2)] * static_cast<double>(0.0833333333333)
                              +in[i*n+(j-2)] * static_cast<double>(-0.0833333333333)
                              +in[(i+2)*n+j] * static_cast<double>(0.0833333333333)
                              +in[(i-2)*n+j] * static_cast<double>(-0.0833333333333)
                              +in[i*n+(j+3)] * static_cast<double>(0.0555555555556)
                              +in[i*n+(j-3)] * static_cast<double>(-0.0555555555556)
                              +in[(i+3)*n+j] * static_cast<double>(0.0555555555556)
                              +in[(i-3)*n+j] * static_cast<double>(-0.0555555555556);
      }
    });
  });
}

void star4(sycl::queue & q, const size_t n, sycl::buffer<double> & d_in, sycl::buffer<double> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::range<2> {n,n}, [=] (sycl::id<2> it) {
      const size_t i = it.get(0);
      const size_t j = it.get(1);
      if ( (4 <= i) && (i < n-4) && (4 <= j) && (j < n-4) ) {
        out[i*n+j] += +in[i*n+(j+1)] * static_cast<double>(0.125)
                              +in[i*n+(j-1)] * static_cast<double>(-0.125)
                              +in[(i+1)*n+j] * static_cast<double>(0.125)
                              +in[(i-1)*n+j] * static_cast<double>(-0.125)
                              +in[i*n+(j+2)] * static_cast<double>(0.0625)
                              +in[i*n+(j-2)] * static_cast<double>(-0.0625)
                              +in[(i+2)*n+j] * static_cast<double>(0.0625)
                              +in[(i-2)*n+j] * static_cast<double>(-0.0625)
                              +in[i*n+(j+3)] * static_cast<double>(0.0416666666667)
                              +in[i*n+(j-3)] * static_cast<double>(-0.0416666666667)
                              +in[(i+3)*n+j] * static_cast<double>(0.0416666666667)
                              +in[(i-3)*n+j] * static_cast<double>(-0.0416666666667)
                              +in[i*n+(j+4)] * static_cast<double>(0.03125)
                              +in[i*n+(j-4)] * static_cast<double>(-0.03125)
                              +in[(i+4)*n+j] * static_cast<double>(0.03125)
                              +in[(i-4)*n+j] * static_cast<double>(-0.03125);
      }
    });
  });
}

void star2(sycl::queue & q, const size_t n, const size_t block_size, sycl::buffer<double> & d_in, sycl::buffer<double> & d_out)
{
  sycl::range<2> global{n,n};
  sycl::range<2> local{block_size,block_size};
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
      const size_t i = it.get_global_id(0);
      const size_t j = it.get_global_id(1);
      if ( (2 <= i) && (i < n-2) && (2 <= j) && (j < n-2) ) {
        out[i*n+j] += +in[i*n+(j+1)] * static_cast<double>(0.25)
                      +in[i*n+(j-1)] * static_cast<double>(-0.25)
                      +in[(i+1)*n+j] * static_cast<double>(0.25)
                      +in[(i-1)*n+j] * static_cast<double>(-0.25)
                      +in[i*n+(j+2)] * static_cast<double>(0.125)
                      +in[i*n+(j-2)] * static_cast<double>(-0.125)
                      +in[(i+2)*n+j] * static_cast<double>(0.125)
                      +in[(i-2)*n+j] * static_cast<double>(-0.125);
      }
    });
  });
}

void star3(sycl::queue & q, const size_t n, const size_t block_size, sycl::buffer<double> & d_in, sycl::buffer<double> & d_out)
{
  sycl::range<2> global{n,n};
  sycl::range<2> local{block_size,block_size};
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
      const size_t i = it.get_global_id(0);
      const size_t j = it.get_global_id(1);
      if ( (3 <= i) && (i < n-3) && (3 <= j) && (j < n-3) ) {
        out[i*n+j] += +in[i*n+(j+1)] * static_cast<double>(0.166666666667)
                      +in[i*n+(j-1)] * static_cast<double>(-0.166666666667)
                      +in[(i+1)*n+j] * static_cast<double>(0.166666666667)
                      +in[(i-1)*n+j] * static_cast<double>(-0.166666666667)
                      +in[i*n+(j+2)] * static_cast<double>(0.0833333333333)
                      +in[i*n+(j-2)] * static_cast<double>(-0.0833333333333)
                      +in[(i+2)*n+j] * static_cast<double>(0.0833333333333)
                      +in[(i-2)*n+j] * static_cast<double>(-0.0833333333333)
                      +in[i*n+(j+3)] * static_cast<double>(0.0555555555556)
                      +in[i*n+(j-3)] * static_cast<double>(-0.0555555555556)
                      +in[(i+3)*n+j] * static_cast<double>(0.0555555555556)
                      +in[(i-3)*n+j] * static_cast<double>(-0.0555555555556);
      }
    });
  });
}

void star4(sycl::queue & q, const size_t n, const size_t block_size, sycl::buffer<double> & d_in, sycl::buffer<double> & d_out)
{
  sycl::range<2> global{n,n};
  sycl::range<2> local{block_size,block_size};
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
      const size_t i = it.get_global_id(0);
      const size_t j = it.get_global_id(1);
      if ( (4 <= i) && (i < n-4) && (4 <= j) && (j < n-4) ) {
        out[i*n+j] += +in[i*n+(j+1)] * static_cast<double>(0.125)
                      +in[i*n+(j-1)] * static_cast<double>(-0.125)
                      +in[(i+1)*n+j] * static_cast<double>(0.125)
                      +in[(i-1)*n+j] * static_cast<double>(-0.125)
                      +in[i*n+(j+2)] * static_cast<double>(0.0625)
                      +in[i*n+(j-2)] * static_cast<double>(-0.0625)
                      +in[(i+2)*n+j] * static_cast<double>(0.0625)
                      +in[(i-2)*n+j] * static_cast<double>(-0.0625)
                      +in[i*n+(j+3)] * static_cast<double>(0.0416666666667)
                      +in[i*n+(j-3)] * static_cast<double>(-0.0416666666667)
                      +in[(i+3)*n+j] * static_cast<double>(0.0416666666667)
                      +in[(i-3)*n+j] * static_cast<double>(-0.0416666666667)
                      +in[i*n+(j+4)] * static_cast<double>(0.03125)
                      +in[i*n+(j-4)] * static_cast<double>(-0.03125)
                      +in[(i+4)*n+j] * static_cast<double>(0.03125)
                      +in[(i-4)*n+j] * static_cast<double>(-0.03125);
      }
    });
  });
}

int main(int argc, char * argv[])
{
  std::cout << "Parallel Research Kernels version " << std::endl;
  std::cout << "C++11/SYCL Stencil execution on 2D grid" << std::endl;

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
          if (block_size < 0) block_size = n;
          if (block_size > n) block_size = n;
      }
      if (block_size && (n % block_size)) {
        throw "ERROR: block size does not evenly divide grid size";
      }

      // stencil radius
      radius = 2;
      if (argc > 4) {
          radius = std::atoi(argv[4]);
      }

      if ( (radius < 2) || (radius > 4) || (2*radius+1 > n) ) {
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
  /// Setup SYCL environment
  //////////////////////////////////////////////////////////////////////

  sycl::queue q{sycl::gpu_selector{}};
  prk::SYCL::print_device_platform(q);

  sycl::range<2> global{n,n};
  sycl::range<2> local{block_size,block_size};

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double stencil_time{0};

  std::vector<double> h_in(n*n,0);
  std::vector<double> h_out(n*n,0);

  try {

    // initialize device buffers from host buffers
    sycl::buffer<double> d_in  { h_in.data(),  h_in.size() };
    sycl::buffer<double> d_out { h_out.data(), h_out.size() };

    q.submit([&](sycl::handler& h) {
      auto in  = d_in.template get_access<sycl::access::mode::read_write>(h);
      h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
          const size_t i = it.get_global_id(0);
          const size_t j = it.get_global_id(1);
          if ((i<n) && (j<n)) {
            in[i*n+j] = static_cast<double>(i+j);
          }
      });
    });
    q.wait();

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) stencil_time = prk::wtime();

      if (block_size) {
          switch (radius) {
              case 2: star2(q, n, block_size, d_in, d_out); break;
              case 3: star3(q, n, block_size, d_in, d_out); break;
              case 4: star4(q, n, block_size, d_in, d_out); break;
          }
          q.wait();
          add(q, n, block_size, d_in);
      } else {
          switch (radius) {
              case 2: star2(q, n, d_in, d_out); break;
              case 3: star3(q, n, d_in, d_out); break;
              case 4: star4(q, n, d_in, d_out); break;
          }
          q.wait();
          add(q, n, d_in);
      }
      q.wait();
    }
    stencil_time = prk::wtime() - stencil_time;
  }
  catch (sycl::exception & e) {
    std::cout << e.what() << std::endl;
    prk::SYCL::print_exception_details(e);
  }
  catch (std::exception & e) {
    std::cout << e.what() << std::endl;
  }
  catch (const char * e) {
    std::cout << e << std::endl;
  }

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
