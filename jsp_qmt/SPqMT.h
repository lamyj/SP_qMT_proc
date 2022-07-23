#ifndef _d36c25a6_f0ec_4184_ba48_a9f0895130b0
#define _d36c25a6_f0ec_4184_ba48_a9f0895130b0

#include <xtensor/xarray.hpp>

namespace SPqMT
{

/// @brief Cost function for the SPqMT fit.
template<typename T>
struct Cost
{
    static double call(double delta, void * data);
};

/// @brief Fit the SPqMT model on a given data point.
xt::xarray<double> fit(xt::xarray<double> data);

}

#include "SPqMT.txx"

#endif // _d36c25a6_f0ec_4184_ba48_a9f0895130b0
