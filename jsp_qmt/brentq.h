#ifndef _58f854bc_2da6_479a_a8bf_75bfcd847db4
#define _58f854bc_2da6_479a_a8bf_75bfcd847db4

#include <type_traits>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_iterator.hpp>

/// @brief Find a function root in a bracketing interval using Brent's method.
template<template<typename> typename Function, typename Data>
typename std::decay<Data>::type
brentq(
    Data const & data,
    double x_lower, double x_upper,
    std::size_t max_iter, double epsabs, double epsrel)
{
    using Container = typename std::decay<Data>::type;
    typename Container::shape_type const shape{
        data.shape().begin(), data.shape().end()-1};
    auto result = xt::empty<typename Container::value_type>(shape);
    
    auto solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
    auto old_handler = gsl_set_error_handler_off();
    
    auto source_it = xt::axis_begin(data), source_end = xt::axis_end(data);
    auto result_it = xt::axis_begin(result);
    gsl_function function{Function<decltype(*source_it)>::call, nullptr};
    while(source_it != source_end)
    {
        auto row = *source_it;
        function.params = &row;
        auto status = gsl_root_fsolver_set(solver, &function, x_lower, x_upper);
        if(status != GSL_SUCCESS)
        {
            *result_it = 0;
        }
        else
        {
            std::size_t iter = 0;
            status = GSL_CONTINUE;
            
            while(status == GSL_CONTINUE && iter < max_iter)
            {
                status = gsl_root_fsolver_iterate(solver);
                if(status != GSL_SUCCESS)
                {
                    break;
                }
                status = gsl_root_test_interval(
                    gsl_root_fsolver_x_lower(solver),
                    gsl_root_fsolver_x_upper(solver),
                    epsabs, epsrel); 
                ++iter;
            }
            
            *result_it = (status == GSL_SUCCESS)?gsl_root_fsolver_root(solver):0;
        }
        
        ++source_it;
        ++result_it;
    }
    
    gsl_set_error_handler(old_handler);
    gsl_root_fsolver_free(solver);
    
    return result;
}

#endif // _58f854bc_2da6_479a_a8bf_75bfcd847db4
