#ifndef _271f4810_cba3_4da6_9864_58174aa8451a
#define _271f4810_cba3_4da6_9864_58174aa8451a

namespace MTsat
{

/// @brief Cost function for the MTsat fit.
template<typename T>
struct Cost
{
    static double call(double delta, void * data);
};

}

#include "MTsat.txx"

#endif // _271f4810_cba3_4da6_9864_58174aa8451a
