#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "SPqMT.h"
#include "super_lorentzian.h"

PYBIND11_MODULE(_SPqMT, m)
{
    xt::import_numpy();
    
    m.def(
        "super_lorentzian",
        pybind11::overload_cast<double, double>(super_lorentzian));
    m.def(
        "super_lorentzian",
        pybind11::overload_cast<double, xt::xarray<double> const &>(
            super_lorentzian));
    m.def("fit", SPqMT::fit);
}
