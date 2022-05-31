import cython
import numpy

from scipy.optimize.cython_optimize cimport brentq

@cython.cdivision
cdef double cost(double delta, void * generic_args):
    cdef double * args = <double *> generic_args
    cdef double E1 = args[0]
    cdef double E2 = args[1]
    cdef double cosFA_RO = args[2]
    cdef double Mz_MT0 = args[3]
    cdef double y = args[4]
    
    cdef double Mz_MTw = (
        ((1-E1) + E1*(1-delta)*(1-E2)) 
        / (1-E1*E2*cosFA_RO*(1-delta))
    )
    cdef double distance = Mz_MTw/Mz_MT0 - y
    return distance

def fit(double[:, ::1] data, double xtol):
    result = numpy.zeros(data.shape[0], dtype=numpy.float)
    for i in range(data.shape[0]):
        # FIXME: error handling
        result[i] = brentq(
            cost, 0., 0.3, &data[i, 0], xtol, 8.881784197001252e-16,
            100, NULL)
    return result
