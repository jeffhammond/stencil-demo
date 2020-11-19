#ifndef PRK_SYCL_HPP
#define PRK_SYCL_HPP

#include "CL/sycl.hpp"

namespace sycl = cl::sycl;

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
            std::cout << "SYCL Platform: " << p.get_info<sycl::info::platform::name>() << std::endl;
            std::cout << "SYCL Device:   " << d.get_info<sycl::info::device::name>() << std::endl;
            //std::cout << "max_work_item_dimensions:" << d.get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
            //auto m = d.get_info<sycl::info::device::max_work_item_sizes>();
            //std::cout << "max_work_item_sizes:" << m[0] << "," << m[1] << "," << m[2] << std::endl;
            //std::cout << "max_work_group_size:" << d.get_info<sycl::info::device::max_work_group_size>() << std::endl;
#endif
        }

        size_t get_max_work_items(const sycl::queue & q) {
            auto d = q.get_device();
            return d.get_info<sycl::info::device::max_work_group_size>();
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
