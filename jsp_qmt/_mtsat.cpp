#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>

#include "brentq.h"
#include "MTsat.h"

void wrap_mtsat(pybind11::module & m)
{
    auto mtsat = m.def_submodule("mtsat");
    mtsat.def(
        "fit",
        [](xt::pyarray<double> const & data, double epsabs) {
            return brentq<MTsat::Cost, decltype(data)>(
                data, 0., 0.3, 100, epsabs, 1e-15); });
}
