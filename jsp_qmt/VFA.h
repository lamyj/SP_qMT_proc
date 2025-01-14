#ifndef _3ecc2fe1_1be3_4d9e_b5d0_6d077f088c78
#define _3ecc2fe1_1be3_4d9e_b5d0_6d077f088c78

#include <type_traits>

namespace VFA
{

template<typename T>
auto linear_fit_E1(T && FA, T && VFA);

template<typename T>
auto linear_fit(T && FA, T && VFA, typename std::decay_t<T>::value_type TR);

template<typename T>
auto non_linear_fit(T && FA, T && VFA, typename std::decay_t<T>::value_type TR);

}

#include "VFA.txx"

#endif // _3ecc2fe1_1be3_4d9e_b5d0_6d077f088c78
