#ifndef _25119426_aaa0_4b01_a184_f42a44d1cc39
#define _25119426_aaa0_4b01_a184_f42a44d1cc39

#include <type_traits>
#include <utility>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_fit.h>
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
    // We are solving Y = scaled_S0 + E1*X, where scaled_S0 = S0*(1-E1)
    // This yields E1 = (y2-y1)/(x2-x1) and scaled_S0=(x2*y1-x1*y2)/(x2-x1)
    // NOTE: if x1==x2, the system is singular
    
    auto const X = xt::eval(VFA / xt::tan(FA));
    auto const Y = xt::eval(VFA / xt::sin(FA));
    
    typename std::decay_t<T>::shape_type const shape(
        VFA.shape().begin(), VFA.shape().end()-1);
    auto E1 = xt::empty<typename std::decay_t<T>::value_type>(shape);
    auto S0 = xt::empty<typename std::decay_t<T>::value_type>(shape);
    
    auto X_it = xt::axis_begin(X), Y_it = xt::axis_begin(Y);
    auto const X_end = xt::axis_end(X);
    auto E1_it = xt::axis_begin(E1), S0_it = xt::axis_begin(S0);
    
    auto const X_stride = X_it->strides()[0];
    auto const Y_stride = Y_it->strides()[0];
    auto const size = FA.shape(1);
    while(X_it != X_end)
    {
        double cov00, cov01, cov11, sumsq;
        auto const status = gsl_fit_linear(
            &(*X_it)(0), X_stride, &(*Y_it)(0), Y_stride, size,
            &(*S0_it)(0), &(*E1_it)(0), &cov00, &cov01, &cov11, &sumsq);
        
        ++X_it; ++Y_it;
        ++E1_it; ++S0_it;
    }
    
    S0 /= (1-E1);
    
    return std::make_pair(E1, S0);
}

template<typename T>
auto linear_fit(T && FA, T && VFA, typename std::decay_t<T>::value_type TR)
{
    auto const [E1, S0] = linear_fit_E1(FA, VFA);
    auto T1 = -TR/xt::log(E1);
    return std::make_pair(xt::eval(T1), S0);
}

template<typename Data>
int signal(gsl_vector const * x, void * data, gsl_vector * f)
{
    auto const & [alpha, S, n] = *reinterpret_cast<Data *>(data);
    auto const S0 = gsl_vector_get(x, 0);
    auto const E1 = gsl_vector_get(x, 1);
    
    for(std::size_t i=0; i<n; ++i)
    {
        auto const cos_a = std::cos(alpha(i));
        auto const sin_a = std::sin(alpha(i));
        auto const Si = S0*((1-E1)*sin_a)/(1-E1*cos_a);
        gsl_vector_set(f, i, Si - (*S)(i));
    }
    return GSL_SUCCESS;
};

template<typename Data>
int jacobian(gsl_vector const * x, void * data, gsl_matrix * J)
{
    auto const & [alpha, S, n] = *reinterpret_cast<Data *>(data);
    auto const S0 = gsl_vector_get(x, 0);
    auto const E1 = gsl_vector_get(x, 1);
    
    for(std::size_t i=0; i<n; ++i)
    {
        auto const cos_a = std::cos(alpha(i));
        auto const sin_a = std::sin(alpha(i));
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
    
    auto fit_parameters = gsl_multifit_nlinear_default_parameters();
    auto workspace = gsl_multifit_nlinear_alloc(
        gsl_multifit_nlinear_trust, &fit_parameters, FA.shape()[1], 2);
    auto old_handler = gsl_set_error_handler_off();
    
    gsl_multifit_nlinear_fdf system;
    using Data = std::tuple<
        T &, decltype(xt::axis_begin(scaled_VFA)), std::size_t>;
    system.f = signal<Data>;
    system.df = jacobian<Data>;
    system.fvv = NULL;
    system.n = FA.shape()[1];
    system.p = 2;
    
    Data params(FA, xt::axis_begin(scaled_VFA), FA.shape()[1]);
    system.params = &params;
    
    auto xtol=1e-8, gtol=1e-8, ftol=1e-8;
    
    // Initial guesses using the linear fit
    auto [E1, S0] = linear_fit_E1(FA, VFA);
    auto E1_it = E1.begin(), S0_it = S0.begin();
    auto guess = gsl_vector_alloc(2);
    
    auto & VFA_it = std::get<1>(params);
    auto VFA_end = xt::axis_end(scaled_VFA);
    while(VFA_it != VFA_end)
    {
        gsl_vector_set(guess, 0, *S0_it);
        gsl_vector_set(guess, 1, *E1_it);
        gsl_multifit_nlinear_winit(guess, nullptr, &system, workspace);
        int info;
        auto const status = gsl_multifit_nlinear_driver(
            1000, xtol, gtol, ftol, nullptr, nullptr, &info, workspace);
        if(status == GSL_SUCCESS)
        {
            *S0_it = gsl_vector_get(workspace->x, 0);
            *E1_it = gsl_vector_get(workspace->x, 1);
        }
        else
        {
            // Keep initial guess
        }
        
        ++E1_it;
        ++S0_it;
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
