import unittest
import numpy
import jsp_qmt

class TestPulse(unittest.TestCase):
    def test_pure_gauss(self):
        value = jsp_qmt.get_pulse_average_and_rms(12e-3, 10, False)
        numpy.testing.assert_allclose(
            value, [0.9957447235048935, 0.995751973473469])
    
    def test_pure_hann(self):
        value = jsp_qmt.get_pulse_average_and_rms(12e-3, 0, True)
        numpy.testing.assert_allclose(
            value, [0.5, 0.6123724356957946])
    
    def test_mixed_gauss_hann(self):
        value = jsp_qmt.get_pulse_average_and_rms(12e-3, 10, True)
        numpy.testing.assert_allclose(
            value, [0.49916428534068447, 0.611745757372692])

if __name__ == "__main__":
    unittest.main()
