#ifndef PRK_UTIL_H
#define PRK_UTIL_H

#include <cstdio>
#include <cstdlib> // atoi, getenv
#include <cstdint>
#include <cfloat>  // FLT_MIN
#include <climits>
#include <cmath>

#include <string>
#include <iostream>
#include <iomanip> // std::setprecision
#include <exception>
#include <vector>
#include <numeric>
#include <algorithm>

#include <chrono>

#define RESTRICT __restrict__

namespace prk {

    template<class I, class T>
    const T reduce(I first, I last, T init) {
#if (defined(__cplusplus) && (__cplusplus >= 201703L)) && !defined(__GNUC__)
        return std::reduce(first, last, init);
#elif (defined(__cplusplus) && (__cplusplus >= 201103L))
        return std::accumulate(first, last, init);
#else
        // unreachable, but preserved as reference implementation
        T r(0);
        for (I i=first; i!=last; ++i) {
            r += *i;
        }
        return r;
#endif
    }

    static inline double wtime(void)
    {
        using t = std::chrono::high_resolution_clock;
        auto c = t::now().time_since_epoch().count();
        auto n = t::period::num;
        auto d = t::period::den;
        double r = static_cast<double>(c)/static_cast<double>(d)*static_cast<double>(n);
        return r;
    }

    template <class T1, class T2>
    static inline auto divceil(T1 numerator, T2 denominator) -> decltype(numerator / denominator) {
        return ( numerator / denominator + (numerator % denominator > 0) );
    }

    bool parse_boolean(const std::string & s)
    {
        if (s=="t" || s=="T" || s=="y" || s=="Y" || s=="1") {
            return true;
        } else {
            return false;
        }

    }

    int get_max_matrix_size(void)
    {
        // std::floor( std::sqrt(INT_MAX) )
        return 46340;
    }

    template <typename T>
    T abs(T x) {
        return (x >= 0 ? x : -x);
    }

    template <>
    float abs(float x) {
        return __builtin_fabsf(x);
    }

    template <>
    double abs(double x) {
        return __builtin_fabs(x);
    }

    template <typename T>
    T sqrt(T x) {
        double y = static_cast<double>(x);
        double z = __builtin_sqrt(y);
        return static_cast<T>(z);
    }

    template <>
    float sqrt(float x) {
        return __builtin_sqrtf(x);
    }

    template <>
    double sqrt(double x) {
        return __builtin_sqrt(x);
    }

    template <typename T>
    T pow(T x, int n) {
        double y = static_cast<double>(x);
        double z = __builtin_pow(y,n);
        return static_cast<T>(z);
    }

    template <>
    double pow(double x, int n) {
        return __builtin_pow(x,n);
    }

    template <>
    float pow(float x, int n) {
        return __builtin_pow(x,n);
    }

} // namespace prk

#endif /* PRK_UTIL_H */
