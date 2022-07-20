#ifndef _443cd058_cfe3_4025_a32c_a1cb9bb2ace1
#define _443cd058_cfe3_4025_a32c_a1cb9bb2ace1

/// @brief C++ wrapper around gsl_function
template<typename Function, typename Data>
class GSLFunction: public gsl_function
{
public:
    /// @brief Actual GSL-compatible call.
    static double call(double x, void * params)
    {
        auto that = reinterpret_cast<GSLFunction<Function, Data>*>(params);
        return that->wrapped_function(x, that->wrapped_data);
    }
    
    /// @brief Wrapped C++ function.
    Function const & wrapped_function;
    
    /// @brief Wrapped C++ data.
    Data & wrapped_data;
    
    GSLFunction(Function const & function, Data & data)
    : wrapped_function(function), wrapped_data(data)
    {
        this->function = GSLFunction::call;
        this->params = this;
    }
};

#endif // _443cd058_cfe3_4025_a32c_a1cb9bb2ace1
