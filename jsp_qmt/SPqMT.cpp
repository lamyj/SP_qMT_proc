#include "SPqMT.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include "GSLFunction.h"

namespace SPqMT
{

xt::xarray<double> fit(xt::xarray<double> data)
{
    static double const xtol = 1e-5;
    
    auto row = xt::row(data, 0);
    GSLFunction<
            decltype(cost<decltype(row)>), decltype(row)
        > function(cost<decltype(row)>, row);
    
    auto solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
    
    auto old_handler = gsl_set_error_handler_off();
    
    auto result = xt::empty<double>(
        xt::xarray<double>::shape_type{data.shape()[0]});
    for(std::size_t row=0; row < data.shape()[0]; ++row)
    {
        function.wrapped_data = xt::row(data, row);
        auto status = gsl_root_fsolver_set(solver, &function, 0., 0.3);
        if(status != GSL_SUCCESS)
        {
            result.unchecked(row) = 0;
            continue;
        }
        
        std::size_t const max_iter = 100;
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
                xtol, 1e-6); 
            ++iter;
        }
        
        result.unchecked(row) = 
            (status == GSL_SUCCESS)?gsl_root_fsolver_root(solver):0;
    }
    
    gsl_set_error_handler(old_handler);
    gsl_root_fsolver_free(solver);
    
    return result;
}

}
