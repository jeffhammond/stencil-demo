#include "prk_util.h"
#include "prk_sycl.h"

void star2(sycl::queue & q, const size_t n, sycl::buffer<double, 2> & d_in, sycl::buffer<double, 2> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    sycl::id<2> dx1(sycl::range<2> {1,0});
    sycl::id<2> dy1(sycl::range<2> {0,1});
    sycl::id<2> dx2(sycl::range<2> {2,0});
    sycl::id<2> dy2(sycl::range<2> {0,2});
    h.parallel_for(sycl::range<2> {n-4,n-4}, sycl::id<2> {2,2}, [=] (sycl::item<2> it) {
        sycl::id<2> xy = it.get_id();
        out[xy] += +in[xy+dx1] * static_cast<double>(0.25)
                   +in[xy-dx1] * static_cast<double>(-0.25)
                   +in[xy+dy1] * static_cast<double>(0.25)
                   +in[xy-dy1] * static_cast<double>(-0.25)
                   +in[xy+dx2] * static_cast<double>(0.125)
                   +in[xy-dx2] * static_cast<double>(-0.125)
                   +in[xy+dy2] * static_cast<double>(0.125)
                   +in[xy-dy2] * static_cast<double>(-0.125);
    });
  });
}

void star3(sycl::queue & q, const size_t n, sycl::buffer<double, 2> & d_in, sycl::buffer<double, 2> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    sycl::id<2> dx1(sycl::range<2> {1,0});
    sycl::id<2> dy1(sycl::range<2> {0,1});
    sycl::id<2> dx2(sycl::range<2> {2,0});
    sycl::id<2> dy2(sycl::range<2> {0,2});
    sycl::id<2> dx3(sycl::range<2> {3,0});
    sycl::id<2> dy3(sycl::range<2> {0,3});
    h.parallel_for(sycl::range<2> {n-6,n-6}, sycl::id<2> {3,3}, [=] (sycl::item<2> it) {
        sycl::id<2> xy = it.get_id();
        out[xy] += +in[xy+dx1] * static_cast<double>(0.166666666667)
                   +in[xy-dx1] * static_cast<double>(-0.166666666667)
                   +in[xy+dy1] * static_cast<double>(0.166666666667)
                   +in[xy-dy1] * static_cast<double>(-0.166666666667)
                   +in[xy+dx2] * static_cast<double>(0.0833333333333)
                   +in[xy-dx2] * static_cast<double>(-0.0833333333333)
                   +in[xy+dy2] * static_cast<double>(0.0833333333333)
                   +in[xy-dy2] * static_cast<double>(-0.0833333333333)
                   +in[xy+dx3] * static_cast<double>(0.0555555555556)
                   +in[xy-dx3] * static_cast<double>(-0.0555555555556)
                   +in[xy+dy3] * static_cast<double>(0.0555555555556)
                   +in[xy-dy3] * static_cast<double>(-0.0555555555556);
    });
  });
}

void star4(sycl::queue & q, const size_t n, sycl::buffer<double, 2> & d_in, sycl::buffer<double, 2> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    sycl::id<2> dx1(sycl::range<2> {1,0});
    sycl::id<2> dy1(sycl::range<2> {0,1});
    sycl::id<2> dx2(sycl::range<2> {2,0});
    sycl::id<2> dy2(sycl::range<2> {0,2});
    sycl::id<2> dx3(sycl::range<2> {3,0});
    sycl::id<2> dy3(sycl::range<2> {0,3});
    sycl::id<2> dx4(sycl::range<2> {4,0});
    sycl::id<2> dy4(sycl::range<2> {0,4});
    h.parallel_for(sycl::range<2> {n-8,n-8}, sycl::id<2> {4,4}, [=] (sycl::item<2> it) {
        sycl::id<2> xy = it.get_id();
        out[xy] += +in[xy+dx1] * static_cast<double>(0.125)
                   +in[xy-dx1] * static_cast<double>(-0.125)
                   +in[xy+dy1] * static_cast<double>(0.125)
                   +in[xy-dy1] * static_cast<double>(-0.125)
                   +in[xy+dx2] * static_cast<double>(0.0625)
                   +in[xy-dx2] * static_cast<double>(-0.0625)
                   +in[xy+dy2] * static_cast<double>(0.0625)
                   +in[xy-dy2] * static_cast<double>(-0.0625)
                   +in[xy+dx3] * static_cast<double>(0.0416666666667)
                   +in[xy-dx3] * static_cast<double>(-0.0416666666667)
                   +in[xy+dy3] * static_cast<double>(0.0416666666667)
                   +in[xy-dy3] * static_cast<double>(-0.0416666666667)
                   +in[xy+dx4] * static_cast<double>(0.03125)
                   +in[xy-dx4] * static_cast<double>(-0.03125)
                   +in[xy+dy4] * static_cast<double>(0.03125)
                   +in[xy-dy4] * static_cast<double>(-0.03125);
    });
  });
}

void nothing(sycl::queue & q, const size_t n, sycl::buffer<double, 2> & d_in, sycl::buffer<double, 2> & d_out)
{
    std::cout << "You are trying to use a stencil that does not exist.\n";
    std::cout << "Please generate the new stencil using the code generator\n";
    std::cout << "and add it to the case-switch in the driver." << std::endl;
    prk::Abort();
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
  /// Setup SYCL environment
  //////////////////////////////////////////////////////////////////////

  sycl::queue q{sycl::gpu_selector{}};
  prk::SYCL::print_device_platform(q);

  auto stencil = nothing;
  switch (radius) {
      case 2: stencil = star2; break;
      case 3: stencil = star3; break;
      case 4: stencil = star4; break;
  }

  size_t padded_n = block_size * prk::divceil(n,block_size);
  sycl::range<2> global{padded_n,padded_n};
  sycl::range<2> local{block_size,block_size};

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double stencil_time{0};

  std::vector<double> h_in(n*n,0);
  std::vector<double> h_out(n*n,0);

  try {

    // initialize device buffers from host buffers
    sycl::buffer<double, 2> d_in  { sycl::range<2> {n, n} };
    sycl::buffer<double, 2> d_out { h_out.data(), sycl::range<2> {n, n} };

    q.submit([&](sycl::handler& h) {
      auto in  = d_in.template get_access<sycl::access::mode::read_write>(h);

      h.parallel_for(sycl::range<2> {n, n}, [=] (sycl::item<2> it) {
          sycl::id<2> xy = it.get_id();
          auto i = it[0];
          auto j = it[1];
          in[xy] = static_cast<double>(i+j);
      });
    });
    q.wait();

    for (int iter = 0; iter<=iterations; iter++) {

      if (iter==1) stencil_time = prk::wtime();

      stencil(q, n, d_in, d_out);
      q.wait();

      q.submit([&](sycl::handler& h) {
        auto in  = d_in.template get_access<sycl::access::mode::read_write>(h);
        // Add constant to solution to force refresh of neighbor data, if any
        h.parallel_for(sycl::range<2> {n, n}, sycl::id<2> {0, 0}, [=] (sycl::item<2> it) {
            sycl::id<2> xy = it.get_id();
            in[xy] += static_cast<double>(1);
        });
      });
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
