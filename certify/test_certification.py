"""Check the public certificate's premises and both transitivity branches.
Run with: python -m unittest certify.test_certification
"""
import math
import unittest

from .core import certify_env
from .finite import FINITE
from toy_env import toy_reveal


class CertificationPremises(unittest.TestCase):
    def test_history_determined_expert_is_certified(self):
        certificate = certify_env(toy_reveal())
        self.assertTrue(certificate.exact)
        self.assertTrue(all(certificate.a2))
        for actual, expected in zip(certificate.H_Gamma, [0, 0, 1, 2]):
            self.assertAlmostEqual(actual, expected)

    def test_private_expert_information_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'A2 fails'):
            certify_env(toy_reveal(leak=0.3))

    def test_nontransitive_instance_returns_bounds(self):
        certificate = certify_env(FINITE['remark_i']())
        self.assertFalse(certificate.exact)
        self.assertTrue(math.isnan(certificate.H_Gamma[1]))
        self.assertAlmostEqual(certificate.H_G[1], 0)
        self.assertAlmostEqual(certificate.H_GammaS[1], math.log2(3))

    def test_a4_failure_does_not_preclude_exactness(self):
        certificate = certify_env(FINITE['rereveal']())
        self.assertTrue(certificate.exact)
        self.assertFalse(all(certificate.a4))
        self.assertTrue(any(g < gs - 0.5 for g, gs in zip(certificate.H_Gamma, certificate.H_GammaS)))


if __name__ == '__main__':
    unittest.main()
