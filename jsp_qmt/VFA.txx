#ifndef _25119426_aaa0_4b01_a184_f42a44d1cc39
#define _25119426_aaa0_4b01_a184_f42a44d1cc39

#include <type_traits>
#include <utility>

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xview.hpp>

namespace VFA
{

template<typename T>
auto linear_fit(T && FA, T && VFA, typename std::decay_t<T>::value_type TR)
{
    // NOTE: xt::eval is not lazy
    auto X = xt::eval(VFA / xt::tan(FA));
    auto Y = xt::eval(VFA / xt::sin(FA));
    
    // We are solving Y = scaled_S0 + E1*X, where scaled_S0 = S0*(1-E1)
    // This yields E1 = (y2-y1)/(x2-x1) and scaled_S0=(x2*y1-x1*y2)/(x2-x1)
    // NOTE: if x1==x2, the system is singular
    
    auto denominator = xt::squeeze(xt::diff(X));
    
    auto E1 = xt::squeeze(xt::diff(Y))/denominator;
    auto T1 = -TR/xt::log(E1);
    
    auto scaled_S0 = 
        (
            xt::view(X, xt::all(), 1) * xt::view(Y, xt::all(), 0)
            - xt::view(X, xt::all(), 0) * xt::view(Y, xt::all(), 1))
        /denominator;
    auto S0 = scaled_S0/(1-E1);
    
    return std::make_pair(xt::eval(T1), xt::eval(S0));
}

}

#endif // _25119426_aaa0_4b01_a184_f42a44d1cc39
