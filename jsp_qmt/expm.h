#ifndef _42991a26_05d8_4b61_bf24_2f2056b46135
#define _42991a26_05d8_4b61_bf24_2f2056b46135

#include <xtensor/xfixed.hpp>

/// @brief Exponential of a 2x2 matrix
xt::xtensor_fixed<double, xt::xshape<2, 2>>
expm_2_2(
    xt::xtensor_fixed<double, xt::xshape<2, 2>> const & A);

#endif // _42991a26_05d8_4b61_bf24_2f2056b46135
