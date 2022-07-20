// cppimport

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

/*
<%
import os
import shlex

here = os.path.dirname(filepath)

extra = ["expm", "SPqMT", "super_lorentzian"]

cfg["compiler_args"] += (
    ["-std=c++17", "-Ofast"]
    + shlex.split(os.environ.get("CXXFLAGS", "")))
cfg["linker_args"] += shlex.split(os.environ.get("LDFLAGS", ""))
cfg["sources"] = [os.path.join(here, f"{x}.cpp") for x in extra]
cfg["dependecies"] = [os.path.join(here, f"{x}.h") for x in extra]
cfg["libraries"] = ["gsl"]
setup_pybind11(cfg)
%>
*/
