#ifndef _ed9cc559_5d76_4b5e_ae91_d89f12c6f205
#define _ed9cc559_5d76_4b5e_ae91_d89f12c6f205

#include <utility>
#include <xtensor/xarray.hpp>

double
super_lorentzian_integrand(double x, std::pair<double, double> const & params);

double super_lorentzian(double T2r, double delta_f);

xt::xarray<double>
super_lorentzian(double T2r, xt::xarray<double> const & delta_f);

#endif // _ed9cc559_5d76_4b5e_ae91_d89f12c6f205
