#include "SPqMT.h"

#include <xtensor/xarray.hpp>

#include "brentq.h"

namespace SPqMT
{

xt::xarray<double> fit(xt::xarray<double> data)
{
    return brentq<Cost, decltype(data)>(data, 0., 0.3, 100, 1e-5, 1e-15);
}

}
