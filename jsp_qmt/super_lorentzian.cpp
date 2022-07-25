#include "super_lorentzian.h"

#include <cmath>
#include <utility>

#include <gsl/gsl_integration.h>
#include <xtensor/xarray.hpp>

thread_local gsl_integration_workspace * workspace =
    gsl_integration_workspace_alloc(1000);

double super_lorentzian_integrand(double x, double T2r, double delta_f)
{
    return 
        T2r/std::fabs((3*std::pow(x, 2)-1))
        * std::exp(-2*std::pow(2*M_PI*delta_f*T2r / (3*std::pow(x, 2)-1), 2));
};

double
super_lorentzian_integrand(double x, void * data)
{
    auto & args = *reinterpret_cast<std::pair<double, double> *>(data);
    auto & [T2r, delta_f] = args;
    return super_lorentzian_integrand(x, T2r, delta_f);
};

double super_lorentzian(double T2r, double delta_f)
{
    auto data = std::make_pair(T2r, delta_f);
    gsl_function function{super_lorentzian_integrand, &data};
    double result, abserr;
    gsl_integration_qags(
        &function, 0., 1., 1e-6, 1e-6, 50, workspace, &result, &abserr);
    return std::sqrt(2/M_PI)*result;
}

xt::xarray<double>
super_lorentzian(double T2r, xt::xarray<double> const & delta_f)
{
    auto data = std::make_pair(T2r, 0.);
    gsl_function function{super_lorentzian_integrand, &data};
    auto result = xt::empty<double>(delta_f.shape());
    double abserr;
    for(std::size_t i=0; i < delta_f.size(); ++i)
    {
        data.second = delta_f.unchecked(i);
        gsl_integration_qags(
            &function, 0., 1., 1.49e-08, 1.49e-08, 50, workspace,
            &result.unchecked(i), &abserr);
    }
    return std::sqrt(2/M_PI)*result;
}
