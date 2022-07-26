#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

#include "expm.h"
#include "super_lorentzian.h"

void wrap_mtsat(pybind11::module &);
void wrap_spqmt(pybind11::module &);
void wrap_vfa(pybind11::module &);

PYBIND11_MODULE(_jsp_qmt, m)
{
    xt::import_numpy();
    
    auto expm_2_2_py = expm_2_2<xt::pyarray<double>>;
    m.def("expm_2_2", expm_2_2_py);
    
    m.def("super_lorentzian_integrand", super_lorentzian_integrand);
    m.def(
        "super_lorentzian",
        pybind11::overload_cast<double, double>(super_lorentzian));
    m.def(
        "super_lorentzian",
        pybind11::overload_cast<double, xt::xarray<double> const &>(
            super_lorentzian));
    
    wrap_mtsat(m);
    wrap_spqmt(m);
    wrap_vfa(m);
}
