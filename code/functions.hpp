#pragma once
#include <numbers>
#include <iostream>
#include "types.hpp"
#include <chrono>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_errno.h>


template <typename T> constexpr T iPow(T x, unsigned int p){
  if (p == 0) return 1;
  T tmp = iPow<T>(x, p/2);
  tmp *= tmp;
  if (p%2 == 1) tmp*=x;
  return tmp;
}

template <typename T> constexpr T f(T x, T temp, T mu, int sigma){
  return 1/(exp((x-mu)/temp)-sigma);
}

inline int mod(int a, int b) {
    return ((a % b) + b) % b;
}

template<typename T = double>
auto digamma(T z) {

    using scalar_type = decltype(std::log(z));
    using real_type = decltype(std::real(std::declval<scalar_type>()));

    constexpr real_type pi = std::numbers::pi_v<real_type>; 

    scalar_type val = 0;

    if (std::real(z) < 0) {
        val -= pi / std::tan(pi * z);
        z = 1.0 - z;
    }

    while (std::real(z) < 6) {
        val -= 1.0 / z;
        z += 1.0;
    }

    constexpr double r1  = -1.0 / 2.0;
    constexpr double r2  = -1.0 / 12.0;
    constexpr double r4  = 1.0 / 120.0;
    constexpr double r6  = -1.0 / 252.0;
    constexpr double r8  = 1.0 / 240.0;
    constexpr double r10 = -1.0 / 132.0;
    constexpr double r12 = 691.0 / 32760.0;
    constexpr double r14 = -1.0 / 12.0;

    val += std::log(z);
    val += r1 / z;

    scalar_type inv_sqr = 1.0 / (z * z);
    val += inv_sqr * (r2 + inv_sqr * 
        (r4 + inv_sqr * 
            (r6 + inv_sqr * 
                (r8 + inv_sqr * 
                    (r10 + inv_sqr * 
                        (r12 + inv_sqr * r14))))));

    return val;
}

template <typename T>
std::complex<double> lngamma(const T& z)
{
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>,
                  "lngamma() only supports double and std::complex<double>");

    if constexpr (std::is_same_v<T, double>) {
        gsl_sf_result lnr;
        int sign;
        int status = gsl_sf_lngamma_sgn_e(z, &lnr, &sign);
        if (status != GSL_SUCCESS) {
            throw std::runtime_error("gsl_sf_lngamma_sgn_e failed: " +
                                      std::string(gsl_strerror(status)));
        }
        // Gamma(x) < 0 for x in (-2,-1), (-4,-3), ... -> ln(Gamma(x)) has imaginary part pi
        double imag = (sign < 0) ? std::numbers::pi : 0.0;
        return std::complex<double>(lnr.val, imag);
    }
    else {
        gsl_sf_result lnr, arg;
        int status = gsl_sf_lngamma_complex_e(z.real(), z.imag(), &lnr, &arg);
        if (status != GSL_SUCCESS) {
            throw std::runtime_error("gsl_sf_lngamma_complex_e failed: " +
                                      std::string(gsl_strerror(status)));
        }
        return std::complex<double>(lnr.val, arg.val);
    }
}
