import warnings
from unittest import TestCase

import numpy as np

from diffprivlib.accountant import BudgetAccountant
from diffprivlib.models import GaussianNB, LogisticRegression, StandardScaler
from diffprivlib.tools import approx_bounds, approx_data_norm
from diffprivlib.tools.bounds import _threshold
from diffprivlib.utils import BudgetError, DiffprivlibCompatibilityWarning, PrivacyLeakWarning, check_random_state


class TestApproxBounds(TestCase):
    def setUp(self):
        self.rng = check_random_state(0)

    def assert_contains_bulk(self, array, bounds, fraction=0.99):
        lower, upper = bounds
        self.assertGreaterEqual(np.mean((array >= lower) & (array <= upper)), fraction)

    def test_distributions(self):
        n = 10000
        arrays = [
            (self.rng.uniform(0, 100, n), 0.99),
            (self.rng.normal(0, 1, n), 0.99),
            (self.rng.standard_cauchy(n), 0.98),
            (-self.rng.uniform(1, 5, n), 0.99),
            (self.rng.normal(0, 1, n) * 1e-7, 0.99),
            (self.rng.randint(-50, 50, n), 0.99),
        ]

        for array, fraction in arrays:
            bounds = approx_bounds(array, epsilon=1, random_state=0)
            self.assertIsInstance(bounds[0], float)
            self.assertIsInstance(bounds[1], float)
            self.assertLess(bounds[0], bounds[1])
            self.assert_contains_bulk(array, bounds, fraction)
            self.assertGreaterEqual(bounds[0], -2 * np.abs(array).max())
            self.assertLessEqual(bounds[1], 2 * np.abs(array).max())

    def test_tight_bounds(self):
        self.assertEqual(approx_bounds(self.rng.uniform(0, 100, 10000), epsilon=1, random_state=0)[1], 128)
        self.assertEqual(approx_bounds(self.rng.uniform(-100, -1, 10000), epsilon=1, random_state=0)[0], -128)

    def test_sign(self):
        positive = self.rng.uniform(1, 5, 1000)
        self.assertGreaterEqual(approx_bounds(positive, random_state=0)[0], 0)
        self.assertLessEqual(approx_bounds(-positive, random_state=0)[1], 0)

    def test_zero_bin(self):
        array = np.concatenate((np.zeros(1000), self.rng.uniform(0, 1, 1000)))
        self.assertEqual(approx_bounds(array, random_state=0)[0], -2.0 ** -64)
        self.assertEqual(approx_bounds(np.zeros(1000), random_state=0), (-2.0 ** -64, 2.0 ** -64))
        self.assertEqual(approx_bounds(np.full(1000, -0.0), random_state=0), (-2.0 ** -64, 2.0 ** -64))
        self.assertEqual(approx_data_norm(np.zeros((1000, 2)), random_state=0), 2.0 ** -64)

    def test_power_of_two_edges(self):
        self.assertEqual(approx_bounds(np.full(1000, 3.0), random_state=0), (2.0, 4.0))
        self.assertEqual(approx_bounds(np.full(1000, 4.0), random_state=0), (4.0, 8.0))
        self.assertEqual(approx_bounds(np.full(1000, -4.0), random_state=0), (-8.0, -4.0))
        self.assertEqual(approx_bounds(np.full(1000, 0.25), random_state=0), (0.25, 0.5))

    def test_exponent_range(self):
        array = np.full(1000, 1e6)
        self.assertEqual(approx_bounds(array, max_exponent=4, random_state=0), (16.0, 32.0))
        self.assertEqual(approx_bounds(-array, max_exponent=4, random_state=0), (-32.0, -16.0))
        self.assertEqual(approx_bounds(array * 1e-12, min_exponent=-4, random_state=0), (-0.0625, 0.0625))

    def test_infinity(self):
        self.assertEqual(approx_bounds(np.full(1000, np.inf), random_state=0), (2.0 ** 64, 2.0 ** 65))
        self.assertEqual(approx_bounds(np.full(1000, -np.inf), random_state=0), (-2.0 ** 65, -2.0 ** 64))

    def test_exponent_limits(self):
        self.assertEqual(approx_bounds(np.full(1000, 1e308), max_exponent=1022, random_state=0),
                         (2.0 ** 1022, 2.0 ** 1023))
        self.assertEqual(approx_bounds(np.full(1000, 5e-324), min_exponent=-1074, random_state=0),
                         (2.0 ** -1074, 2.0 ** -1073))
        self.assertEqual(approx_data_norm(np.zeros((1000, 2)), min_exponent=-1074, random_state=0), 2.0 ** -1074)

    def test_integer_and_list_input(self):
        self.assertEqual(approx_bounds(np.full(1000, 2 ** 62, dtype=np.int64), random_state=0), (2.0 ** 62, 2.0 ** 63))
        self.assertEqual(approx_bounds([3] * 1000, random_state=0), (2.0, 4.0))
        self.assertEqual(approx_data_norm(np.full((1000, 2), 3, dtype=np.int32), random_state=0), 8.0)

    def test_2d(self):
        array = np.column_stack((self.rng.normal(0, 1, 10000), self.rng.uniform(0, 100, 10000)))
        lower, upper = approx_bounds(array, epsilon=2, random_state=0)
        self.assertEqual(lower.shape, (2,))
        self.assertEqual(upper.shape, (2,))

        for i in range(2):
            self.assert_contains_bulk(array[:, i], (lower[i], upper[i]))

        single_lower, single_upper = approx_bounds(array[:, 1:], epsilon=1, random_state=0)
        self.assertEqual(single_lower.shape, (1,))
        self.assertEqual(single_upper[0], 128)

    def test_gaussian_nb(self):
        X = np.column_stack((self.rng.normal(0, 1, 2000), self.rng.uniform(0, 100, 2000)))
        y = self.rng.randint(2, size=2000)
        bounds = approx_bounds(X, epsilon=1, random_state=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Bounds", category=PrivacyLeakWarning)
            GaussianNB(epsilon=1, bounds=bounds, random_state=0).fit(X, y)
            StandardScaler(epsilon=1, bounds=bounds, random_state=0).fit(X)

    def test_logistic_regression(self):
        X = self.rng.normal(0, 1, (2000, 3))
        y = (X[:, 0] > 0).astype(int)
        data_norm = approx_data_norm(X, epsilon=1, random_state=0)
        self.assertIsInstance(data_norm, float)
        self.assertGreaterEqual(np.mean(np.linalg.norm(X, axis=1) <= data_norm), 0.99)
        self.assertLessEqual(data_norm, 2 * np.linalg.norm(X, axis=1).max())

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Data norm has not been specified", category=PrivacyLeakWarning)
            LogisticRegression(epsilon=1, data_norm=data_norm, random_state=0).fit(X, y)

    def test_accountant(self):
        array = self.rng.normal(0, 1, (1000, 3))
        acc = BudgetAccountant()
        approx_bounds(array, epsilon=0.6, accountant=acc, random_state=0)
        self.assertEqual(acc.total(), (0.6, 0))
        approx_data_norm(array, epsilon=0.4, accountant=acc, random_state=0)
        self.assertAlmostEqual(acc.total()[0], 1.0)
        self.assertEqual(len(acc), 2)

        with BudgetAccountant(1.5) as acc:
            approx_bounds(array, epsilon=1, random_state=0)
            with self.assertRaises(BudgetError):
                approx_bounds(array, epsilon=1, random_state=0)
            with self.assertRaises(BudgetError):
                approx_data_norm(array, epsilon=1, random_state=0)
            self.assertEqual(acc.total(), (1, 0))

    def test_random_state(self):
        array = self.rng.normal(0, 1, (200, 2))
        kwargs = {"epsilon": 2, "confidence": 0.5}
        self.assertEqual(approx_data_norm(array, random_state=3, **kwargs),
                         approx_data_norm(array, random_state=3, **kwargs))
        first = approx_bounds(array, random_state=3, **kwargs)
        second = approx_bounds(array, random_state=3, **kwargs)
        self.assertTrue(np.array_equal(first, second))

    def test_threshold(self):
        epsilon, n_bins, confidence = 0.5, 131, 1 - 1e-6
        expected = np.log(1 / (1 - confidence ** (1 / (n_bins - 1)))) / epsilon
        self.assertAlmostEqual(_threshold(epsilon, n_bins, confidence), expected, places=6)

    def test_insufficient_data(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            approx_bounds(np.ones(10), epsilon=0.1, random_state=0)
        with self.assertRaisesRegex(ValueError, "threshold"):
            approx_data_norm(np.ones((10, 2)), epsilon=0.1, random_state=0)

    def test_failure_spends_budget(self):
        acc = BudgetAccountant()
        with self.assertRaises(ValueError):
            approx_bounds(np.ones(10), epsilon=0.1, accountant=acc, random_state=0)
        with self.assertRaises(ValueError):
            approx_data_norm(np.ones((10, 2)), epsilon=0.2, accountant=acc, random_state=0)
        self.assertAlmostEqual(acc.total()[0], 0.3)

    def test_failure_in_one_column_spends_budget(self):
        spread = np.tile(2.0 ** np.arange(-32, 33), 3)
        array = np.column_stack((np.full(spread.size, 3.0), spread))
        acc = BudgetAccountant()

        with self.assertRaises(ValueError):
            approx_bounds(array, epsilon=4, accountant=acc, random_state=0)
        self.assertEqual(acc.total(), (4, 0))
        self.assertEqual(approx_bounds(array[:, 0], epsilon=2, random_state=0), (2.0, 4.0))

    def test_bad_epsilon(self):
        for epsilon in (0, -1):
            with self.assertRaises(ValueError):
                approx_bounds(np.ones(100), epsilon=epsilon)
            with self.assertRaises(ValueError):
                approx_data_norm(np.ones((100, 2)), epsilon=epsilon)

    def test_bad_confidence(self):
        for confidence in (0, 1, -0.5, 1.5):
            with self.assertRaises(ValueError):
                approx_bounds(np.ones(100), confidence=confidence)
            with self.assertRaises(ValueError):
                approx_data_norm(np.ones((100, 2)), confidence=confidence)
        with self.assertRaises(TypeError):
            approx_bounds(np.ones(100), confidence="0.5")

    def test_bad_exponents(self):
        for min_exponent, max_exponent in ((5, 4), (-1075, 0), (0, 1023)):
            with self.assertRaises(ValueError):
                approx_bounds(np.ones(100), min_exponent=min_exponent, max_exponent=max_exponent)
            with self.assertRaises(ValueError):
                approx_data_norm(np.ones((100, 2)), min_exponent=min_exponent, max_exponent=max_exponent)
        with self.assertRaises(TypeError):
            approx_bounds(np.ones(100), min_exponent=1.5)

    def test_bad_array(self):
        with self.assertRaises(ValueError):
            approx_bounds([])
        with self.assertRaises(ValueError):
            approx_bounds(np.ones((2, 2, 2)))
        with self.assertRaises(ValueError):
            approx_bounds(np.array([1.0, np.nan] * 500))
        with self.assertRaises(ValueError):
            approx_data_norm(np.ones(100))
        with self.assertRaises(ValueError):
            approx_data_norm(np.ones((0, 2)))
        with self.assertRaises(ValueError):
            approx_data_norm(np.array([[1.0, np.nan]] * 500))

    def test_unused_args(self):
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            approx_bounds(np.ones(1000), axis=0, random_state=0)
