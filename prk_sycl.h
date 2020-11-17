#ifndef PRK_SYCL_HPP
#define PRK_SYCL_HPP

#include "CL/sycl.hpp"

namespace sycl = cl::sycl;

#if 0
typedef float prk_float;
#else
typedef double prk_float;
#endif

namespace prk {

    // There seems to be an issue with the clang CUDA/HIP toolchains not having
    // std::abort() available
    void Abort(void) {
#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HCC)
        abort();
#else
        std::abort();
#endif
    }

    namespace SYCL {

        void print_device_platform(const sycl::queue & q) {
#if ! ( defined(TRISYCL) || defined(__HIPSYCL__) )
            auto d = q.get_device();
            auto p = d.get_platform();
            std::cout << "SYCL Device:   " << d.get_info<sycl::info::device::name>() << std::endl;
            std::cout << "SYCL Platform: " << p.get_info<sycl::info::platform::name>() << std::endl;
#endif
        }

        bool has_fp64(const sycl::queue & q) {
#if 0
            return false;
#else
            return true;
#endif
        }

        void print_exception_details(sycl::exception & e) {
            std::cout << e.what() << std::endl;
#ifdef __COMPUTECPP__
            std::cout << e.get_file_name() << std::endl;
            std::cout << e.get_line_number() << std::endl;
            std::cout << e.get_description() << std::endl;
            std::cout << e.get_cl_error_message() << std::endl;
            std::cout << e.get_cl_code() << std::endl;
#endif
        }

    } // namespace SYCL

} // namespace prk

#endif // PRK_SYCL_HPP
