from unittest import TestCase

import numpy as np

from diffprivlib.tools.utils import mean
from diffprivlib.utils import PrivacyLeakWarning, BudgetError


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
        self.assertIsNotNone(mean(a, bounds=(0, 1)))

    def test_no_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = mean(a, epsilon=1)
        self.assertIsNotNone(res)

    def test_bad_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            mean(a, epsilon=1, bounds=(0, -1))

    def test_missing_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = mean(a, epsilon=1, bounds=None)
        self.assertIsNotNone(res)

    def test_large_epsilon(self):
        a = np.random.random(1000)
        res = float(np.mean(a))
        res_dp = mean(a, epsilon=1, bounds=(0, 1))

        self.assertAlmostEqual(res, res_dp, delta=0.01)

    def test_large_epsilon_axis(self):
        a = np.random.random((1000, 5))
        res = np.mean(a, axis=0)
        res_dp = mean(a, epsilon=1, bounds=(0, 1), axis=0)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i], res_dp[i], delta=0.01)

    def test_nan(self):
        a = np.random.random((5, 5))
        a[2, 2] = np.nan

        res = mean(a, bounds=(0, 1))
        self.assertTrue(np.isnan(res))

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant(1.5, 0)

        a = np.random.random((1000, 5))
        mean(a, epsilon=1, bounds=(0, 1), accountant=acc)
        self.assertEqual((1.0, 0), acc.total())

        with acc:
            with self.assertRaises(BudgetError):
                mean(a, epsilon=1, bounds=(0, 1))
