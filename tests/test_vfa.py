import unittest
import numpy
import jsp_qmt

class TestVFA(unittest.TestCase):
    def test_linear_fit_VFA_2(self):
        S0 = numpy.array([1000, 1500, 800])[:,None]
        T1 = numpy.array([1.2, 0.9, 1.6])[:,None]
        
        flip_angles = numpy.radians([10, 30])[None,:]
        TR = 30e-3
        
        E1 = numpy.exp(-TR/T1)
        S = S0*(1-E1)*numpy.sin(flip_angles)/(1-E1*numpy.cos(flip_angles))
        
        T1_hat, S0_hat = jsp_qmt.vfa.linear_fit(flip_angles, S, TR)
        
        numpy.testing.assert_allclose(T1_hat, T1.squeeze())
        numpy.testing.assert_allclose(S0_hat, S0.squeeze())
    
    def test_linear_fit_VFA_3(self):
        S0 = numpy.array([1000, 1500, 800])[:,None]
        T1 = numpy.array([1.2, 0.9, 1.6])[:,None]
        
        flip_angles = numpy.radians([10, 20, 30])[None,:]
        TR = 30e-3
        
        E1 = numpy.exp(-TR/T1)
        S = S0*(1-E1)*numpy.sin(flip_angles)/(1-E1*numpy.cos(flip_angles))
        
        T1_hat, S0_hat = jsp_qmt.vfa.linear_fit(flip_angles, S, TR)
        
        numpy.testing.assert_allclose(T1_hat, T1.squeeze())
        numpy.testing.assert_allclose(S0_hat, S0.squeeze())
    
    def test_non_linear_fit_2(self):
        S0 = numpy.array([1000, 1500, 800])[:,None]
        T1 = numpy.array([1.2, 0.9, 1.6])[:,None]
        
        flip_angles = numpy.radians([10, 30])[None,:]
        TR = 30e-3
        
        E1 = numpy.exp(-TR/T1)
        S = S0*(1-E1)*numpy.sin(flip_angles)/(1-E1*numpy.cos(flip_angles))
        
        T1_hat, S0_hat = jsp_qmt.vfa.non_linear_fit(flip_angles, S, TR)
        
        numpy.testing.assert_allclose(T1_hat, T1.squeeze())
        numpy.testing.assert_allclose(S0_hat, S0.squeeze())
    
    def test_non_linear_fit_3(self):
        S0 = numpy.array([1000, 1500, 800])[:,None]
        T1 = numpy.array([1.2, 0.9, 1.6])[:,None]
        
        flip_angles = numpy.radians([10, 20, 30])[None,:]
        TR = 30e-3
        
        E1 = numpy.exp(-TR/T1)
        S = S0*(1-E1)*numpy.sin(flip_angles)/(1-E1*numpy.cos(flip_angles))
        
        T1_hat, S0_hat = jsp_qmt.vfa.non_linear_fit(flip_angles, S, TR)
        
        numpy.testing.assert_allclose(T1_hat, T1.squeeze())
        numpy.testing.assert_allclose(S0_hat, S0.squeeze())

if __name__ == "__main__":
    unittest.main()
