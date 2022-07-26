import numpy
import scipy.integrate
import unittest

import jsp_qmt

class TestSuperLorentzian(unittest.TestCase):
    def test_integrand(self):
        x, T2r, delta_f = 0.5, 10e-6, 100
        
        v1 = jsp_qmt.super_lorentzian_integrand(x, T2r, delta_f)
        v2 = (
            T2r/numpy.abs((3*x**2 - 1))
            * numpy.exp(-2*(2*numpy.pi*delta_f*T2r / (3*x**2 - 1))**2))
        
        numpy.testing.assert_allclose(v1, v2)
    
    def test_integral_scalar(self):
        T2r, delta_f = 10e-6, 100
        
        v1 = jsp_qmt.super_lorentzian(T2r, delta_f)
        v2 = scipy.integrate.quad(
            lambda x: (
                numpy.sqrt(2/numpy.pi)
                * jsp_qmt.super_lorentzian_integrand(x, T2r, delta_f)),
            0., 1.)[0]
        
        numpy.testing.assert_allclose(v1, v2, rtol=5e-6)
    
    def test_integral_array(self):
        T2r, delta_f_array = 10e-6, [100, 200]
        
        v1 = jsp_qmt.super_lorentzian(T2r, delta_f_array)
        v2 = [
            scipy.integrate.quad(
                lambda x: (
                    numpy.sqrt(2/numpy.pi)
                    * jsp_qmt.super_lorentzian_integrand(x, T2r, delta_f)),
                0., 1.)[0]
            for delta_f in delta_f_array]
        
        numpy.testing.assert_allclose(v1, v2, rtol=5e-6)

if __name__ == "__main__":
    unittest.main()
