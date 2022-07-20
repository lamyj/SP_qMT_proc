#ifndef _c4dbca66_b7e2_4a00_8c88_bbf21a1ffc60
#define _c4dbca66_b7e2_4a00_8c88_bbf21a1ffc60

#include "SPqMT.h"

#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include "xtensor-blas/xlinalg.hpp"

#include "expm.h"

namespace SPqMT
{

template<typename T>
double cost(double F, T const & args)
{
    using Double2x2 = xt::xtensor_fixed<double, xt::xshape<2, 2>>;
    using Double2 = xt::xtensor_fixed<double, xt::xshape<2>>;
    
    double Wb = args.unchecked(0), Wf = args.unchecked(1),
        FAro = args.unchecked(2), R1 = args.unchecked(3), R = args.unchecked(4),
        Ts = args.unchecked(5), Tm = args.unchecked(6), Tr = args.unchecked(7),
        y = args.unchecked(8);
    
    auto const R1r = R1;
    auto const R1f = R1;
    auto const f = F/(1+F);
    
    // Non-variable
    Double2x2 const Rl = {{-R1f-R*F, R}, {R*F, -R1r-R}};
    Double2 const Meq = {1-f, f};
    auto const A = R1f*R1r + R1f*R + R1r*R*F;
    auto const D = A + (R1f+R*F)*Wb + (R1r+R)*Wf + Wb*Wf;
    auto const Es = expm_2_2(Rl*Ts);
    auto const Er = expm_2_2(Rl*Tr);
    Double2x2 const C = {{std::cos(FAro*M_PI/180.), 0.}, {0., 1.}};
    static Double2x2 const I = {{1., 0.}, {0., 1.}};
    Double2x2 const W = {{-Wf, 0.}, {0., -Wb}};
    
    // MTw
    Double2 const Mss = {(1.-f)*(A+R1f*Wb)/D, f*(A+R1r*Wf)/D};
    auto const Em = expm_2_2((Rl+W)*Tm);
    using xt::linalg::dot;
    auto const Mz = dot(
        xt::linalg::inv(I - dot(dot(Es, Em), dot(Er, C))),
        dot(dot(dot(Es, Em), I-Er) + I-Es, Meq)
            + dot(dot(Es, I-Em), Mss));
    
    // MT0
    Double2 const MssN = {(1-f)*A/A, f*A/A};
    auto const EmN = expm_2_2(Rl*Tm);
    auto const MzN = dot(
        xt::linalg::inv(I - dot(dot(Es, EmN), dot(Er, C))),
        dot(dot(dot(Es, EmN), I-Er) + I-Es, Meq)
            + dot(dot(Es, I-EmN), MssN));
    
    return Mz[0]/MzN[0] - y;
}

}

#endif // _c4dbca66_b7e2_4a00_8c88_bbf21a1ffc60
