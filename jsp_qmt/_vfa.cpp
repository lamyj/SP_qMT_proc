#include <pybind11/pybind11.h>
#include <xtensor-python/pyarray.hpp>

#include "VFA.h"

void wrap_vfa(pybind11::module & m)
{
    auto vfa = m.def_submodule("vfa");
    
    auto linear_fit_py = VFA::linear_fit<xt::pyarray<double>>;
    vfa.def("linear_fit", linear_fit_py);
    
    auto non_linear_fit_py = VFA::non_linear_fit<xt::pyarray<double>>;
    vfa.def("non_linear_fit", non_linear_fit_py);
}
