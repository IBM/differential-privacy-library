import numpy as np
from unittest import TestCase

import pytest

from diffprivlib.mechanisms import Snapping
from diffprivlib.utils import global_seed


class TestSnapping(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Snapping

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Snapping, DPMechanism))

    def test_neg_sensitivity(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, sensitivity=-1, lower=0, upper=1000)

    def test_str_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, sensitivity="1", lower=0, upper=1000)

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, sensitivity=0, lower=0, upper=1000)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, sensitivity=1, lower=0, upper=1000)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), sensitivity=1, lower=0, upper=1000)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, sensitivity=1, lower=0, upper=1000)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", sensitivity=1, lower=0, upper=1000)

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, sensitivity=1, lower=0, upper=1000))
        self.assertIn(".Snapping(", repr_)

    def test_epsilon(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=1000)
        self.assertIsNotNone(mech.randomise(1))

    def test_non_numeric(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=1000)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_zero_median_prob(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=1000)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_effective_epsilon_high_sensitivity(self):
        mech = self.mech(epsilon=1, sensitivity=100, lower=0, upper=1)

        self.assertLess(mech.effective_epsilon(), 1.0)

    def test_effective_epsilon_one_sensitivity(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=1)

        self.assertGreater(mech.effective_epsilon(), 1.0)

    def test_effective_epsilon_zero_sensitivity(self):
        mech = self.mech(epsilon=1, sensitivity=0, lower=0, upper=1)

        self.assertTrue(mech.effective_epsilon(), float('inf'))

    def test_rounding_power_of_two(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=1)

        self.assertEqual(mech._lambda, 1)

    def test_rounding_not_power_of_two(self):
        mech = self.mech(epsilon=3, sensitivity=1, lower=0, upper=1)

        self.assertEqual(mech._lambda, 0.5)

    def test_within_bounds(self):
        mech = self.mech(epsilon=1, sensitivity=1, lower=0, upper=1)
        vals = []

        for i in range(1000):
            vals.append(mech.randomise(0.5))

        vals = np.array(vals)

        self.assertTrue(np.all(vals >= 0))
        self.assertTrue(np.all(vals <= 1))

    def test_neighbours_prob(self):
        epsilon = 1
        runs = 10000
        mech = self.mech(epsilon=epsilon, sensitivity=1, lower=0, upper=1000)
        count = [0, 0]

        for i in range(runs):
            val0 = mech.randomise(0)
            if val0 <= 0:
                count[0] += 1

            val1 = mech.randomise(1)
            if val1 <= 0:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)

