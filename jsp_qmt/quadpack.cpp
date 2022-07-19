#include "quadpack.h"

#include <functional>
#include <iostream>
#include <string>

#include <dlfcn.h>
#include <pybind11/pybind11.h>

thread_local Quadpack * Quadpack::_instance = nullptr;

double
Quadpack
::quadpack_wrapper(double *x)
{
    return Quadpack::instance().function(*x);
}

Quadpack
::Quadpack()
{
    auto _quadpack = pybind11::module::import("scipy.integrate._quadpack");
    auto const path = _quadpack.attr("__file__").cast<std::string>();
    
    this->_library = dlopen(path.c_str(), RTLD_LAZY);
    if(!this->_library)
    {
        std::cerr << "Could not open " << path << std::endl;
        return;
    }
    
    dlerror();
    this->_qagse = reinterpret_cast<qagse_t>(dlsym(this->_library, "dqagse_"));
    char const * error = dlerror();
    if(error)
    {
        std::cerr << "Cannot load dqagse: " << error << std::endl;;
        dlclose(this->_library);
        this->_library = nullptr;
        this->_qagse = nullptr;
    }
}

Quadpack
::~Quadpack()
{
    std::cout << "Destroying lib" << std::endl;
    this->_qagse = nullptr;
    if(this->_library != nullptr)
    {
        dlclose(this->_library);
    }
}

Quadpack &
Quadpack
::instance()
{
    if(Quadpack::_instance == nullptr)
    {
        Quadpack::_instance = new Quadpack();
    }
    return *Quadpack::_instance;
}

Quadpack::qagse_t
Quadpack
::qagse() const
{
    return this->_qagse;
}

double qagse(double low, double high, double epsabs, double epsrel)
{
    int limit=50;

    double result;
    double abserr;
    int neval, ier;
    double alist[50], blist[50], rlist[50], elist[50];
    int iord[50];
    int last;
    Quadpack::instance().qagse()(
        Quadpack::quadpack_wrapper, &low, &high, &epsabs, &epsrel, &limit,
        &result,
        &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);

    return result;
}
