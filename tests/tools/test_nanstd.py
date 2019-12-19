from unittest import TestCase

import numpy as np

from diffprivlib.tools.utils import nanstd
from diffprivlib.utils import PrivacyLeakWarning, global_seed


class TestNanStd(TestCase):
    def test_not_none(self):
        mech = nanstd
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanstd(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(nanstd(a, range=1))

    def test_no_range(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            nanstd(a, epsilon=1)

    def test_negative_range(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            nanstd(a, epsilon=1, range=-1)

    def test_missing_range(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanstd(a, 1, None)
        self.assertIsNotNone(res)

    def test_large_epsilon(self):
        global_seed(12345)
        a = np.random.random(1000)
        res = float(np.std(a))
        res_dp = nanstd(a, epsilon=1, range=1)

        self.assertAlmostEqual(res, res_dp, delta=0.01)

    def test_large_epsilon_axis(self):
        global_seed(12345)
        a = np.random.random((1000, 5))
        res = np.std(a, axis=0)
        res_dp = nanstd(a, epsilon=1, range=1, axis=0)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i], res_dp[i], delta=0.01)

    def test_nan(self):
        a = np.random.random((5, 5))
        a[2, 2] = np.nan

        res = nanstd(a, range=1)
        self.assertFalse(np.isnan(res))

