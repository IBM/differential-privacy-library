import warnings

import numpy as np
from unittest import TestCase

import pytest

from diffprivlib.mechanisms import LaplaceBoundedDomain


class TestLaplaceBoundedDomain(TestCase):
    def setup_method(self, method):
        self.mech = LaplaceBoundedDomain

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(LaplaceBoundedDomain, DPMechanism))

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=0, lower=0, upper=2)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), delta=0, sensitivity=1, lower=0, upper=1)

        for i in range(1000):
            self.assertEqual(mech.randomise(0.5), 0.5)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, delta=0, sensitivity=1, lower=0, upper=1)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", delta=0, sensitivity=1, lower=0, upper=1)

    def test_wrong_bounds(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, delta=0, sensitivity=1, lower=3, upper=1)

        with self.assertRaises(TypeError):
            self.mech(epsilon=1, delta=0, sensitivity=1, lower="0", upper="2")

    def test_non_numeric(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_zero_median_prob(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1, random_state=0)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0.5))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.5, delta=0.1)

    def test_neighbors_prob(self):
        epsilon = 1
        runs = 1000
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1, random_state=0)

        count0 = (np.array([mech.randomise(0) for _ in range(runs)]) <= 0.5).sum()
        count1 = (np.array([mech.randomise(1) for _ in range(runs)]) <= 0.5).sum()

        self.assertGreater(count0, count1)
        self.assertLessEqual(count0 / runs, np.exp(epsilon) * count1 / runs + 0.1)

    def test_within_bounds(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        vals = []

        for i in range(1000):
            vals.append(mech.randomise(0.5))

        vals = np.array(vals)

        self.assertTrue(np.all(vals >= 0))
        self.assertTrue(np.all(vals <= 1))

    def test_semi_inf_domain_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), delta=0, sensitivity=1, lower=0.0, upper=float("inf"))

        with warnings.catch_warnings(record=True) as w:
            self.assertIsNotNone(mech.randomise(0))

        self.assertFalse(w, "Warning thrown for LaplaceBoundedDomain")

    def test_random_state(self):
        mech1 = self.mech(epsilon=1, sensitivity=1, lower=0, upper=4, random_state=42)
        mech2 = self.mech(epsilon=1, sensitivity=1, lower=0, upper=4, random_state=42)
        self.assertEqual([mech1.randomise(0) for _ in range(100)], [mech2.randomise(0) for _ in range(100)])

        self.assertNotEqual([mech1.randomise(0)] * 100, [mech1.randomise(0) for _ in range(100)])

        rng = np.random.RandomState(0)
        mech1 = self.mech(epsilon=1, sensitivity=1, lower=0, upper=4, random_state=rng)
        mech2 = self.mech(epsilon=1, sensitivity=1, lower=0, upper=4, random_state=rng)
        self.assertNotEqual([mech1.randomise(0) for _ in range(100)], [mech2.randomise(0) for _ in range(100)])

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1))
        self.assertIn(".LaplaceBoundedDomain(", repr_)

    def test_bias(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        self.assertGreater(mech.bias(0), 0.0)
        self.assertLess(mech.bias(1), 0.0)

    def test_variance(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        self.assertGreater(mech.variance(0), 0.0)

    def test_effective_epsilon(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=10)
        self.assertLessEqual(mech.effective_epsilon(), 1.0)

    def test_effective_epsilon_same(self):
        mech = self.mech(epsilon=1, delta=0, sensitivity=1, lower=0, upper=1)
        self.assertEqual(mech.effective_epsilon(), 1.0)

    def test_effective_epsilon_nonzero_delta(self):
        mech = self.mech(epsilon=1, delta=0.5, sensitivity=1, lower=0, upper=10)
        self.assertIsNone(mech.effective_epsilon())
