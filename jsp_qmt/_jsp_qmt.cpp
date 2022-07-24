#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "brentq.h"
#include "expm.h"
#include "MTsat.h"
#include "SPqMT.h"
#include "super_lorentzian.h"

PYBIND11_MODULE(_jsp_qmt, m)
{
    xt::import_numpy();
    
    m.def("expm_2_2", expm_2_2<xt::pyarray<double>>);
    m.def(
        "super_lorentzian",
        pybind11::overload_cast<double, double>(super_lorentzian));
    m.def(
        "super_lorentzian",
        pybind11::overload_cast<double, xt::xarray<double> const &>(
            super_lorentzian));
    
    auto mtsat = m.def_submodule("mtsat");
    mtsat.def(
        "fit",
        [](xt::xarray<double> const & data, double epsabs) {
            return brentq<MTsat::Cost, decltype(data)>(
                data, 0., 0.3, 100, epsabs, 1e-15); });
    
    auto spqmt = m.def_submodule("spqmt");
    spqmt.def(
        "fit",
        [](xt::xarray<double> const & data) {
            return brentq<SPqMT::Cost, decltype(data)>(
                data, 0., 0.3, 100, 1e-5, 1e-15); });
}
