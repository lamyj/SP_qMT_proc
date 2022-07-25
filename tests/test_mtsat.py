import numpy
import unittest

import jsp_qmt

class TestMTSat(unittest.TestCase):
    def test_fit(self):
        T1 = 1
        delta = 0.15
        
        TR, TR1, FA = 28e-3, 12e-3, numpy.radians(6.)
        
        E1 = numpy.exp(-TR/T1)
        E2 = numpy.exp(-(TR-TR1) / T1)
        cosFA = numpy.cos(FA)
        Mz_MT0 = (1-E1*E2) / (1-E1*E2*cosFA)
        Mz_MTw = ((1-E1) + E1*(1-delta)*(1-E2)) / (1-E1*E2*cosFA*(1-delta))
        
        signal_ratio = Mz_MTw/Mz_MT0
        data = numpy.array([[E1, E2, cosFA, Mz_MT0, signal_ratio]])
        
        value = jsp_qmt.mtsat.fit(data, 1e-7)
        numpy.testing.assert_allclose(value, delta)

if __name__ == "__main__":
    unittest.main()
