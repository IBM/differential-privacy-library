from unittest import TestCase

import numpy as np

from diffprivlib.tools.utils import var


class TestVar(TestCase):
    def test_not_none(self):
        mech = var
        self.assertIsNotNone(mech)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            var(a)

    def test_no_range(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            var(a, 1)

    def test_negative_range(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            var(a, 1, -1)

    def test_missing_range(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            var(a, 1, None)

    def test_large_epsilon(self):
        a = np.random.random(1000)
        res = float(np.var(a))
        res_dp = var(a, epsilon=1, range=1)

        self.assertAlmostEqual(res, res_dp, delta=0.01)

    def test_large_epsilon_axis(self):
        a = np.random.random((1000, 5))
        res = np.var(a, axis=0)
        res_dp = var(a, epsilon=1, range=1, axis=0)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i], res_dp[i], delta=0.01)
