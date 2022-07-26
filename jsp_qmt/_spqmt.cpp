#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>

#include "brentq.h"
#include "SPqMT.h"

void wrap_spqmt(pybind11::module & m)
{
    auto spqmt = m.def_submodule("spqmt");
    spqmt.def(
        "fit",
        [](xt::pyarray<double> const & data) {
            return brentq<SPqMT::Cost, decltype(data)>(
                data, 0., 0.3, 100, 1e-5, 1e-15); });
}
