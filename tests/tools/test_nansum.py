from unittest import TestCase

import numpy as np

from diffprivlib.tools.utils import nansum
from diffprivlib.utils import PrivacyLeakWarning, BudgetError


class TestNansum(TestCase):
    def test_not_none(self):
        mech = nansum
        self.assertIsNotNone(mech)

    def test_no_params(self):
        a = np.array([1, 2, 3])
        with self.assertWarns(PrivacyLeakWarning):
            res = nansum(a)
        self.assertIsNotNone(res)

    def test_no_epsilon(self):
        a = np.array([1, 2, 3])
        self.assertIsNotNone(nansum(a, bounds=(1, 3)))

    def test_no_bounds(self):
        a = np.array([1, 2, 3, np.nan])
        with self.assertWarns(PrivacyLeakWarning):
            res = nansum(a, epsilon=1)
        self.assertIsNotNone(res)

    def test_mis_ordered_bounds(self):
        a = np.array([1, 2, 3, np.nan])
        with self.assertRaises(ValueError):
            nansum(a, epsilon=1, bounds=(1, 0))

    def test_missing_bounds(self):
        a = np.array([1, 2, 3, np.nan])
        with self.assertWarns(PrivacyLeakWarning):
            res = nansum(a, epsilon=1, bounds=None)
        self.assertIsNotNone(res)

    def test_inf_epsilon(self):
        a = np.random.random(1000)
        res = float(np.nansum(a))
        res_dp = nansum(a, epsilon=float("inf"), bounds=(0, 1))

        self.assertAlmostEqual(res, res_dp)

    def test_large_epsilon(self):
        a = np.random.random(1000)
        res = float(np.nansum(a))
        res_dp = nansum(a, epsilon=5, bounds=(0, 1))

        self.assertAlmostEqual(res, res_dp, delta=0.01 * res)

    def test_array_like(self):
        self.assertIsNotNone(nansum([1, 2, 3, np.nan], bounds=(1, 3)))
        self.assertIsNotNone(nansum((1, 2, 3, np.nan), bounds=(1, 3)))

    def test_axis(self):
        a = np.random.random((1000, 5))
        res_dp = nansum(a, epsilon=1, axis=0, bounds=(0, 1))
        self.assertEqual(res_dp.shape, (5,))

    def test_clipped_output(self):
        a = np.random.random((10,))

        for i in range(100):
            self.assertTrue(0 <= nansum(a, epsilon=1e-5, bounds=(0, 1)) <= 10)

    def test_int_output(self):
        a = np.random.random(1000) * 10
        res_int = nansum(a, dtype=int, bounds=(0, 10))
        self.assertIsInstance(res_int, int)

        res = np.nansum(a, dtype=int)
        res_inf = nansum(a, epsilon=float("inf"), dtype=int, bounds=(0, 10))
        self.assertEqual(res, res_inf)

    def test_nan(self):
        a = np.random.random((5, 5))
        a[2, 2] = np.nan

        res = nansum(a, bounds=(0, 1))
        self.assertFalse(np.isnan(res))

        with self.assertWarns(PrivacyLeakWarning):
            res = nansum(a)
        self.assertFalse(np.isnan(res))

        a = np.array([np.nan] * 10)
        res = nansum(a, epsilon=float("inf"), bounds=(0, 1))
        self.assertEqual(0, res)

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant(1.5, 0)

        a = np.random.random((1000, 5))
        nansum(a, epsilon=1, bounds=(0, 1), accountant=acc)
        self.assertEqual((1.0, 0), acc.total())

        with acc:
            with self.assertRaises(BudgetError):
                nansum(a, epsilon=1, bounds=(0, 1))

    def test_replacing_a_value_by_nan_is_not_distinguishable(self):
        a = np.array([10] * 50)
        b = np.array([10] * 49 + [np.nan])
        res_a = np.array([nansum(a, epsilon=1, bounds=(5, 10), random_state=i) for i in range(1000)])
        res_b = np.array([nansum(b, epsilon=1, bounds=(5, 10), random_state=1000 + i) for i in range(1000)])

        for event_a, event_b in (res_a > 495, res_b > 495), (res_a <= 495, res_b <= 495):
            p_a, p_b = event_a.mean(), event_b.mean()
            self.assertLessEqual(p_a, np.e * p_b + 0.05)
            self.assertLessEqual(p_b, np.e * p_a + 0.05)

    def test_replacing_a_negative_value_by_nan_is_not_distinguishable(self):
        a = np.array([-10] * 50)
        b = np.array([-10] * 49 + [np.nan])
        res_a = np.array([nansum(a, epsilon=1, bounds=(-10, -5), random_state=i) for i in range(1000)])
        res_b = np.array([nansum(b, epsilon=1, bounds=(-10, -5), random_state=1000 + i) for i in range(1000)])

        for event_a, event_b in (res_a > -495, res_b > -495), (res_a <= -495, res_b <= -495):
            p_a, p_b = event_a.mean(), event_b.mean()
            self.assertLessEqual(p_a, np.e * p_b + 0.05)
            self.assertLessEqual(p_b, np.e * p_a + 0.05)

    def test_replacing_an_integer_value_by_nan_is_not_distinguishable(self):
        a = np.array([10] * 50)
        b = np.array([10] * 49 + [np.nan])
        res_a = np.array([nansum(a, epsilon=1, bounds=(5, 10), dtype=int, random_state=i) for i in range(1000)])
        res_b = np.array([nansum(b, epsilon=1, bounds=(5, 10), dtype=int, random_state=1000 + i) for i in range(1000)])

        for event_a, event_b in (res_a > 495, res_b > 495), (res_a <= 495, res_b <= 495):
            p_a, p_b = event_a.mean(), event_b.mean()
            self.assertLessEqual(p_a, np.e * p_b + 0.05)
            self.assertLessEqual(p_b, np.e * p_a + 0.05)
