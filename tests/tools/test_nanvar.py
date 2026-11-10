from unittest import TestCase

import numpy as np

from diffprivlib.tools.utils import nanvar
from diffprivlib.utils import PrivacyLeakWarning, BudgetError, check_random_state


class TestNanVar(TestCase):
    def test_not_none(self):
        mech = nanvar
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanvar(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(nanvar(a, bounds=(0, 1)))

    def test_no_bounds(self):
        a = np.array([1, 2, 3, np.nan])
        with self.assertWarns(PrivacyLeakWarning):
            nanvar(a, epsilon=1)

    def test_bad_bounds(self):
        a = np.array([1, 2, 3, np.nan])
        with self.assertRaises(ValueError):
            nanvar(a, epsilon=1, bounds=(0, -1))

    def test_missing_bounds(self):
        a = np.array([1, 2, 3, np.nan])
        with self.assertWarns(PrivacyLeakWarning):
            res = nanvar(a, 1, None)
        self.assertIsNotNone(res)

    def test_large_epsilon(self):
        a = np.random.random(1000)
        res = float(np.var(a))
        res_dp = nanvar(a, epsilon=5, bounds=(0, 1))

        self.assertAlmostEqual(res, res_dp, delta=0.01)

    def test_large_epsilon_axis(self):
        a = np.random.random((1000, 5))
        res = np.var(a, axis=0)
        res_dp = nanvar(a, epsilon=15, bounds=(0, 1), axis=0)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i], res_dp[i], delta=0.01)

    def test_array_like(self):
        self.assertIsNotNone(nanvar([1, 2, 3, np.nan], bounds=(1, 3)))
        self.assertIsNotNone(nanvar((1, 2, 3, np.nan), bounds=(1, 3)))

    def test_clipped_output(self):
        a = np.random.random((10,))
        rng = check_random_state(0)

        for i in range(100):
            self.assertTrue(0 <= nanvar(a, epsilon=1e-3, bounds=(0, 1), random_state=rng) <= 1)

    def test_nan(self):
        a = np.random.random((5, 5))
        a[2, 2] = np.nan

        res = nanvar(a, bounds=(0, 1))
        self.assertFalse(np.isnan(res))

        with self.assertWarns(PrivacyLeakWarning):
            res = nanvar(a)
        self.assertFalse(np.isnan(res))

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant(1.5, 0)

        a = np.random.random((1000, 5))
        nanvar(a, epsilon=1, bounds=(0, 1), accountant=acc)
        self.assertEqual((1.0, 0), acc.total())

        with acc:
            with self.assertRaises(BudgetError):
                nanvar(a, epsilon=1, bounds=(0, 1))

    def test_neighbouring_arrays_are_not_distinguishable(self):
        a = np.array([0, 1] + [np.nan] * 98)
        b = np.array([1, 1] + [np.nan] * 98)
        res_a = np.array([nanvar(a, epsilon=1, bounds=(0, 1), random_state=i) for i in range(1000)])
        res_b = np.array([nanvar(b, epsilon=1, bounds=(0, 1), random_state=1000 + i) for i in range(1000)])

        for event_a, event_b in (res_a > 0.1, res_b > 0.1), (res_a <= 0.1, res_b <= 0.1):
            p_a, p_b = event_a.mean(), event_b.mean()
            self.assertLessEqual(p_a, np.e * p_b + 0.05)
            self.assertLessEqual(p_b, np.e * p_a + 0.05)

    def test_all_nan_slice(self):
        a = np.array([np.nan, np.nan, np.nan])

        for i in range(20):
            res = nanvar(a, epsilon=1, bounds=(0, 1), random_state=i)
            self.assertTrue(0 <= res <= 0.25)

    def test_replacing_a_value_by_nan_is_not_distinguishable(self):
        a = np.array([0, 1] + [np.nan] * 98)
        b = np.array([0, 1, 1] + [np.nan] * 97)
        res_a = np.array([nanvar(a, epsilon=1, bounds=(0, 1), random_state=i) for i in range(1000)])
        res_b = np.array([nanvar(b, epsilon=1, bounds=(0, 1), random_state=1000 + i) for i in range(1000)])

        for event_a, event_b in (res_a > 0.237, res_b > 0.237), (res_a <= 0.237, res_b <= 0.237):
            p_a, p_b = event_a.mean(), event_b.mean()
            self.assertLessEqual(p_a, np.e * p_b + 0.05)
            self.assertLessEqual(p_b, np.e * p_a + 0.05)
