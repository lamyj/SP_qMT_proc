#ifndef _0917d93b_07b4_470b_a570_a6009520bece
#define _0917d93b_07b4_470b_a570_a6009520bece

#include <functional>

/// @brief Singleton wrapping the Quadpack library, loaded from scipy.
class Quadpack
{
public:
    using integrand = double (*)(double*);
    using qagse_t = void (*)( 
        integrand f, double *a, double *b, double *epsabs, double *epsrel,
        int *limit, double *result, double *abserr, int *neval, int *ier,
        double *alist, double *blist, double *rlist, double *elist,
        int *iord, int *last);
    
    static double quadpack_wrapper(double *x);
    
    std::function<double(double)> function;
    
    static Quadpack & instance();
    
    qagse_t qagse() const;
    
private:
    thread_local static Quadpack * _instance;
    void * _library;
    qagse_t _qagse;
    
    Quadpack();
    ~Quadpack();
};

/**
 * @brief Wrapper for the qagse method of Quadpack, using the integrand defined
 *        in the Quadpack singleton.
 */
double
qagse(
    double low, double high, double epsabs=1.49e-08, double epsrel=1.49e-08);

#endif // _0917d93b_07b4_470b_a570_a6009520bece
