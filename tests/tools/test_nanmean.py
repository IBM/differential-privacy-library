from unittest import TestCase

import numpy as np

from diffprivlib.tools.utils import nanmean
from diffprivlib.utils import PrivacyLeakWarning


class TestNanMean(TestCase):
    def test_not_none(self):
        mech = nanmean
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanmean(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(nanmean(a, range=1))

    def test_no_range(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanmean(a, epsilon=1)
        self.assertIsNotNone(res)

    def test_negative_range(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            nanmean(a, epsilon=1, range=-1)

    def test_missing_range(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanmean(a, epsilon=1, range=None)
        self.assertIsNotNone(res)

    def test_large_epsilon(self):
        a = np.random.random(1000)
        res = float(np.mean(a))
        res_dp = nanmean(a, epsilon=1, range=1)

        self.assertAlmostEqual(res, res_dp, delta=0.01)

    def test_large_epsilon_axis(self):
        a = np.random.random((1000, 5))
        res = np.mean(a, axis=0)
        res_dp = nanmean(a, epsilon=1, range=1, axis=0)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i], res_dp[i], delta=0.01)

    def test_nan(self):
        a = np.random.random((5, 5))
        a[2, 2] = np.nan

        res = nanmean(a, range=1)
        self.assertFalse(np.isnan(res))
