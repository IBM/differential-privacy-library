from unittest import TestCase

import numpy as np

from diffprivlib.tools.utils import std
from diffprivlib.utils import PrivacyLeakWarning, BudgetError, check_random_state


class TestStd(TestCase):
    def test_not_none(self):
        mech = std
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = std(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(std(a, bounds=(0, 1)))

    def test_no_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            std(a, epsilon=1)

    def test_bad_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            std(a, epsilon=1, bounds=(0, -1))

    def test_missing_bounds(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = std(a, 1, None)
        self.assertIsNotNone(res)

    def test_large_epsilon(self):
        a = np.random.random(1000)
        res = float(np.std(a))
        res_dp = std(a, epsilon=5, bounds=(0, 1))

        self.assertAlmostEqual(res, res_dp, delta=0.01)

    def test_large_epsilon_axis(self):
        a = np.random.random((1000, 5))
        res = np.std(a, axis=0)
        res_dp = std(a, epsilon=15, bounds=(0, 1), axis=0)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i], res_dp[i], delta=0.01)

    def test_array_like(self):
        self.assertIsNotNone(std([1, 2, 3], bounds=(1, 3)))
        self.assertIsNotNone(std((1, 2, 3), bounds=(1, 3)))

    def test_clipped_output(self):
        a = np.random.random((10,))
        rng = check_random_state(0)

        for i in range(100):
            self.assertTrue(0 <= std(a, epsilon=1e-3, bounds=(0, 1), random_state=rng) <= 1)

    def test_nan(self):
        a = np.random.random((5, 5))
        a[2, 2] = np.nan

        res = std(a, bounds=(0, 1))
        self.assertTrue(np.isnan(res))

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant(1.5, 0)

        a = np.random.random((1000, 5))
        std(a, epsilon=1, bounds=(0, 1), accountant=acc)
        self.assertEqual((1.0, 0), acc.total())

        with acc:
            with self.assertRaises(BudgetError):
                std(a, epsilon=1, bounds=(0, 1))
