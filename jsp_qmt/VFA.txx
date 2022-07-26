#ifndef _25119426_aaa0_4b01_a184_f42a44d1cc39
#define _25119426_aaa0_4b01_a184_f42a44d1cc39

#include <type_traits>
#include <utility>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multifit_nlinear.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xview.hpp>

namespace VFA
{

template<typename T>
auto linear_fit_E1(T && FA, T && VFA)
{
    // NOTE: xt::eval is not lazy
    auto X = xt::eval(VFA / xt::tan(FA));
    auto Y = xt::eval(VFA / xt::sin(FA));
    
    // We are solving Y = scaled_S0 + E1*X, where scaled_S0 = S0*(1-E1)
    // This yields E1 = (y2-y1)/(x2-x1) and scaled_S0=(x2*y1-x1*y2)/(x2-x1)
    // NOTE: if x1==x2, the system is singular
    
    auto denominator = xt::squeeze(xt::diff(X));
    
    auto E1 = xt::squeeze(xt::diff(Y))/denominator;
    
    auto scaled_S0 = 
        (
            xt::view(X, xt::all(), 1) * xt::view(Y, xt::all(), 0)
            - xt::view(X, xt::all(), 0) * xt::view(Y, xt::all(), 1))
        /denominator;
    auto S0 = scaled_S0/(1-E1);
    
    return std::make_pair(xt::eval(E1), xt::eval(S0));
}

template<typename T>
auto linear_fit(T && FA, T && VFA, typename std::decay_t<T>::value_type TR)
{
    auto [E1, S0] = linear_fit_E1(FA, VFA);
    auto T1 = -TR/xt::log(E1);
    return std::make_pair(xt::eval(T1), S0);
}

template<typename Data>
int signal(gsl_vector const * x, void * data, gsl_vector * f)
{
    auto [alpha, S, n] = *reinterpret_cast<Data *>(data);
    auto const S0 = gsl_vector_get(x, 0);
    auto const E1 = gsl_vector_get(x, 1);
    
    for(std::size_t i=0; i<n; ++i)
    {
        auto const cos_a = std::cos((*alpha)(i));
        auto const sin_a = std::sin((*alpha)(i));
        auto const Si = S0*((1-E1)*sin_a)/(1-E1*cos_a);
        gsl_vector_set(f, i, Si - (*S)(i));
    }
    return GSL_SUCCESS;
};

template<typename Data>
int jacobian(gsl_vector const * x, void * data, gsl_matrix * J)
{
    auto [alpha, S, n] = *reinterpret_cast<Data *>(data);
    auto const S0 = gsl_vector_get(x, 0);
    auto const E1 = gsl_vector_get(x, 1);

    for(std::size_t i=0; i<n; ++i)
    {
        auto const cos_a = std::cos((*alpha)(i));
        auto const sin_a = std::sin((*alpha)(i));
        auto const df_dS0 = ((1-E1)*sin_a) / (1 - E1*cos_a);
        auto const df_dE1 = S0*(sin_a*(cos_a-1)) / std::pow(1 - E1*cos_a, 2);
        gsl_matrix_set(J, i, 0, df_dS0);
        gsl_matrix_set(J, i, 1, df_dE1);
    }
    return GSL_SUCCESS;
};

template<typename T>
auto non_linear_fit(T && FA, T && VFA, typename std::decay_t<T>::value_type TR)
{
    // We are solving Y = f(S0, E1) = S0 ((1-E1) sin α) / (1 - E1 cos α)
    // The partial derivatives of f are
    // ∂f/∂S0 = ((1-E1) sin α) / (1 - E1 cos α)
    // ∂f/∂E1 = S0 ( -sin α (1 - E1 cos α) + (1-E1) sin α cos α ) / ((1 - E1 cos α)²)
    //        = S0 (sin α ((1-E1) cos α - 1 + E1 cos α) ) / ((1 - E1 cos α)²)
    //        = S0 (sin α (cos α - 1)) / ((1 - E1 cos α)²)
    
    auto fit = gsl_multifit_nlinear_trust;
    auto fit_parameters = gsl_multifit_nlinear_default_parameters();
    auto workspace = gsl_multifit_nlinear_alloc(
        fit, &fit_parameters, FA.shape()[1], 2);
    auto old_handler = gsl_set_error_handler_off();
    
    gsl_multifit_nlinear_fdf system;
    using Data = std::tuple<
        decltype(xt::axis_begin(FA)), decltype(xt::axis_begin(VFA)), std::size_t>;
    system.f = signal<Data>;
    system.df = jacobian<Data>;
    system.fvv = NULL;
    system.n = FA.shape()[1];
    system.p = 2;
    auto xtol=1e-8, gtol=1e-8, ftol=1e-8;
    
    // Initial guesses using the linear fit
    auto [E1, S0] = linear_fit_E1(FA, VFA);
    auto E1_it = E1.begin(), S0_it = S0.begin();
    auto guess = gsl_vector_alloc(2);
    
    auto FA_it = xt::axis_begin(FA), FA_end = xt::axis_end(FA);
    auto VFA_it = xt::axis_begin(VFA);
    while(FA_it != FA_end)
    {
        Data params(FA_it, VFA_it, FA.shape()[1]);
        system.params = &params;
        
        gsl_vector_set(guess, 0, *S0_it);
        gsl_vector_set(guess, 1, *E1_it);
        gsl_multifit_nlinear_winit(guess, NULL, &system, workspace);
        int info;
        auto status = gsl_multifit_nlinear_driver(
            1000, xtol, gtol, ftol, NULL, NULL, &info, workspace);
        if(status == GSL_SUCCESS)
        {
            *S0_it = gsl_vector_get(workspace->x, 0);
            *E1_it = gsl_vector_get(workspace->x, 1);
        }
        else
        {
            *S0_it = 0.;
            *E1_it = 0.;
        }
        
        ++E1_it;
        ++S0_it;
        ++FA_it;
        ++VFA_it;
    }
    
    gsl_vector_free(guess);
    
    gsl_set_error_handler(old_handler);
    gsl_multifit_nlinear_free(workspace);
    
    auto T1 = -TR/xt::log(E1);
    return std::make_pair(xt::eval(T1), S0);
}

}

#endif // _25119426_aaa0_4b01_a184_f42a44d1cc39
