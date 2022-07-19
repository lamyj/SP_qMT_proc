/*
Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "brentq.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include <xtensor/xarray.hpp>

double
brentq(
    std::function<double(double, xt::xarray<double> const &)> function,
    xt::xarray<double> const & data,
    double xa, double xb, double xtol, double rtol, int max_iter)
{
    auto xpre = xa, xcur = xb;
    auto fpre = function(xpre, data);
    auto fcur = function(xcur, data);
    
    if(fpre*fcur > 0)
    {
        return 0.;
    }
    else if(fpre == 0)
    {
        return xpre;
    }
    else if(fcur == 0)
    {
        return xcur;
    }
    
    double xblk = 0., fblk = 0., spre = 0., scur = 0.;
    for(int i = 0; i < max_iter; ++i)
    {
        if(fpre != 0 && fcur != 0 && std::signbit(fpre) != std::signbit(fcur))
        {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if(std::fabs(fblk) < std::fabs(fcur))
        {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
    
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
    
        /* the tolerance is 2*delta */
        auto const delta = (xtol + rtol*std::fabs(xcur))/2.;
        auto const sbis = (xblk - xcur)/2.;
        if(fcur == 0 || std::fabs(sbis) < delta)
        {
            return xcur;
        }
    
        if(std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre))
        {
            double stry;
            if(xpre == xblk)
            {
                /* interpolate */
                stry = -fcur*(xcur - xpre)/(fcur - fpre);
            }
            else
            {
                /* extrapolate */
                auto const dpre = (fpre - fcur)/(xpre - xcur);
                auto const dblk = (fblk - fcur)/(xblk - xcur);
                stry = -fcur*(fblk*dblk - fpre*dpre)/(dblk*dpre*(fblk - fpre));
            }
            if(
                2*std::fabs(stry) < std::min(
                    std::fabs(spre), 3*std::fabs(sbis) - delta))
            {
                /* good short step */
                spre = scur;
                scur = stry;
            }
            else
            {
                /* bisect */
                spre = sbis;
                scur = sbis;
            }
        }
        else
        {
            /* bisect */
            spre = sbis;
            scur = sbis;
        }
    
        xpre = xcur;
        fpre = fcur;
        if(std::fabs(scur) > delta)
        {
            xcur += scur;
        }
        else
        {
            xcur += (sbis > 0 ? delta : -delta);
        }
    
        fcur = function(xcur, data);
    }
    return xcur;
}
