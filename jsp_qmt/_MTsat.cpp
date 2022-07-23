#include <pybind11/pybind11.h>
#include <xtensor/xarray.hpp>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "brentq.h"

namespace MTsat
{

/// @brief Cost function for the MTsat fit.
template<typename T>
struct Cost
{
    static double call(double delta, void * data)
    {
        T const & args = *reinterpret_cast<T*>(data);
        double E1 = args.unchecked(0), E2 = args.unchecked(1),
            cosFA_RO = args.unchecked(2), Mz_MT0 = args.unchecked(3),
            y = args.unchecked(4);
        auto const Mz_MTw =
            ((1-E1) + E1*(1-delta)*(1-E2)) / (1-E1*E2*cosFA_RO*(1-delta));
        return Mz_MTw/Mz_MT0 - y;
    }
};

/// @brief Fit the MTsat model on a given data point.
xt::xarray<double> fit(xt::xarray<double> const & data, double epsabs)
{
    return brentq<Cost, decltype(data)>(data, 0., 0.3, 100, epsabs, 1e-15);
}

}

PYBIND11_MODULE(_MTsat, m)
{
    xt::import_numpy();
    m.def("fit", MTsat::fit);
}
