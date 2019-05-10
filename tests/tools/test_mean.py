from unittest import TestCase

import numpy as np

from diffprivlib.tools.utils import mean
from diffprivlib.utils import PrivacyLeakWarning


class TestMean(TestCase):
    def test_not_none(self):
        mech = mean
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = mean(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(mean(a, range=1))

    def test_no_range(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = mean(a, epsilon=1)
        self.assertIsNotNone(res)

    def test_negative_range(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            mean(a, epsilon=1, range=-1)

    def test_missing_range(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = mean(a, epsilon=1, range=None)
        self.assertIsNotNone(res)

    def test_large_epsilon(self):
        a = np.random.random(1000)
        res = float(np.mean(a))
        res_dp = mean(a, epsilon=1, range=1)

        self.assertAlmostEqual(res, res_dp, delta=0.01)

    def test_large_epsilon_axis(self):
        a = np.random.random((1000, 5))
        res = np.mean(a, axis=0)
        res_dp = mean(a, epsilon=1, range=1, axis=0)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i], res_dp[i], delta=0.01)
