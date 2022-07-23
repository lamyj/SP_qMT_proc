import numpy
import scipy
import unittest

import jsp_qmt._jsp_qmt

class TestExpm(unittest.TestCase):
    def test_diagonal(self):
        M = numpy.diag([42., 42.])
        e1 = scipy.linalg.expm(M)
        e2 = jsp_qmt._jsp_qmt.expm_2_2(M)
        numpy.testing.assert_allclose(e1, e2)
    
    def test_defective(self):
        M = numpy.array([[3., 1.], [0., 3.]])
        e1 = scipy.linalg.expm(M)
        e2 = jsp_qmt._jsp_qmt.expm_2_2(M)
        numpy.testing.assert_allclose(e1, e2)
    
    def test_conjugate_eigenvalues(self):
        v1 = numpy.array(
            [numpy.cos(numpy.radians(37)), numpy.sin(numpy.radians(37))])
        v2 = numpy.array(
            [numpy.sin(numpy.radians(37)), -numpy.cos(numpy.radians(37))])
        M = numpy.vstack((v1, -v2))
        e1 = scipy.linalg.expm(M)
        e2 = jsp_qmt._jsp_qmt.expm_2_2(M)
        numpy.testing.assert_allclose(e1, e2)
    
    def test_opposite_eigenvalues(self):
        v1 = numpy.array(
            [numpy.cos(numpy.radians(37)), numpy.sin(numpy.radians(37))])
        v2 = numpy.array(
            [numpy.sin(numpy.radians(37)), -numpy.cos(numpy.radians(37))])
        D = numpy.diag([12, -12])
        P = numpy.vstack((v1, v2))
        M = P @ D @ numpy.linalg.inv(P)
        e1 = scipy.linalg.expm(M)
        e2 = jsp_qmt._jsp_qmt.expm_2_2(M)
        numpy.testing.assert_allclose(e1, e2)
    
    def test_one_null_eigenvalue(self):
        M = numpy.array([[3., 0.], [0., 0.]])
        e1 = scipy.linalg.expm(M)
        e2 = jsp_qmt._jsp_qmt.expm_2_2(M)
        numpy.testing.assert_allclose(e1, e2)
    
    def test_random(self):
        numpy.random.seed(0)

        matrices = numpy.random.random((1000, 2, 2))

        e1 = numpy.empty(matrices.shape)
        for i, M in enumerate(matrices):
            e1[i] = scipy.linalg.expm(M)
        
        e2 = numpy.empty(matrices.shape)
        for i, M in enumerate(matrices):
            e2[i] = jsp_qmt._jsp_qmt.expm_2_2(M)
        numpy.testing.assert_allclose(e1, e2)

if __name__ == "__main__":
    unittest.main()
