import scipy
import numpy
import unittest

import jsp_qmt

class TestSPqMT(unittest.TestCase):
    def test_fit(self):
        gamma = 267.513 * 1e6
        R1 = 1.
        F = 0.23
        R1fT2f, T2r, R = 0.022, 10.0e-6, 19.0
        
        FAsat = numpy.radians(560)
        delta_f = 6000
        Tm = 12e-3
        FWHM = 9
        HannApo = True
        
        FAro = numpy.radians(6)
        Ts = 2e-3
        Tp = 100e-6
        TR = 28e-3
        
        average, rms = jsp_qmt.get_pulse_average_and_rms(Tm, FWHM, HannApo)
        B1peak_nom = FAsat / (gamma*average*Tm)
        w1RMS_nom = gamma*B1peak_nom * rms
        G = jsp_qmt.super_lorentzian(T2r, delta_f)
        Wb = (numpy.pi * w1RMS_nom**2 * G)
        
        Wf = (w1RMS_nom/(2*numpy.pi))**2 / R1fT2f / (delta_f)**2 * R1
        
        R1r = R1f = R1
        f = F/(1+F)
        
        # non-variable
        Rl = numpy.array([[-R1f-R*F, R],[R*F, -R1r-R]])
        Meq = numpy.array([1-f, f])
        A = R1f*R1r + R1f*R + R1r*R*F
        D = A + (R1f+R*F)*Wb + (R1r+R)*Wf + Wb*Wf
        Es = scipy.linalg.expm(Rl*Ts)
        Er = scipy.linalg.expm(Rl*TR)
        C = numpy.diag([numpy.cos(FAro), 1.0])
        I = numpy.eye(2)
        W = numpy.array([[-Wf, 0],[0, -Wb]])
        
        # MTw
        Mss = 1/D*numpy.array([(1-f)*(A+R1f*Wb), f*(A+R1r*Wf)])   
        Em = scipy.linalg.expm((Rl+W)*Tm)
        Mz = (
            scipy.linalg.inv(I - Es @ Em @ Er @ C) 
            @ ( (Es @ Em @ (I-Er) + (I-Es)) @ Meq + Es @ (I-Em) @ Mss ))
        
        # MT0
        MssN = 1/A * numpy.array([(1-f)*A, f*A])
        EmN = scipy.linalg.expm(Rl*Tm)
        MzN = (
            scipy.linalg.inv(I - Es @ EmN @ Er @ C) 
            @ ( (Es @ EmN @ (I - Er) + (I-Es)) @ Meq + Es @ ( I-EmN ) @ MssN ))
        
        
        signal_ratio = Mz[0]/MzN[0]
        data = numpy.array([[Wb, Wf, FAro, R1, R, Ts, Tm, TR, signal_ratio]])
        value = jsp_qmt.spqmt.fit(data)
        
        numpy.testing.assert_allclose(value, F)

if __name__ == "__main__":
    unittest.main()
