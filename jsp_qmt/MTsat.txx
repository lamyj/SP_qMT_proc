#ifndef _1e7d149b_86d0_4f25_9e1b_e681c531e3af
#define _1e7d149b_86d0_4f25_9e1b_e681c531e3af

#include "MTsat.h"

namespace MTsat
{

template<typename T>
double
Cost<T>
::call(double delta, void * data)
{
    T const & args = *reinterpret_cast<T*>(data);
    double E1 = args.unchecked(0), E2 = args.unchecked(1),
        cosFA_RO = args.unchecked(2), Mz_MT0 = args.unchecked(3),
        y = args.unchecked(4);
    auto const Mz_MTw =
        ((1-E1) + E1*(1-delta)*(1-E2)) / (1-E1*E2*cosFA_RO*(1-delta));
    return Mz_MTw/Mz_MT0 - y;
}

}

#endif // _1e7d149b_86d0_4f25_9e1b_e681c531e3af
