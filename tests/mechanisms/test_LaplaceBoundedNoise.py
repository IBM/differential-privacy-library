import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import LaplaceBoundedNoise
from diffprivlib.utils import global_seed


class TestLaplaceBoundedNoise(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = LaplaceBoundedNoise()

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(LaplaceBoundedNoise, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_no_sensitivity(self):
        self.mech.set_epsilon_delta(1, 0.1)
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_zero_sensitivity(self):
        self.mech.set_sensitivity(0).set_epsilon_delta(1, 0.1)

        for i in range(1000):
            self.assertAlmostEqual(self.mech.randomise(1), 1)

    def test_no_epsilon(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_neg_epsilon(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(-1, 0.1)

    def test_inf_epsilon(self):
        self.mech.set_sensitivity(1).set_epsilon_delta(float("inf"), 0.1)

        for i in range(1000):
            self.assertEqual(self.mech.randomise(1), 1)

    def test_zero_epsilon(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(0, 0.1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon_delta(1+2j, 0.25)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon_delta("Two", 0.25)

    def test_no_delta(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon(1)

    def test_large_delta(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(1, 0.6)

    def test_non_numeric(self):
        self.mech.set_sensitivity(1).set_epsilon_delta(1, 0.1)
        with self.assertRaises(TypeError):
            self.mech.randomise("Hello")

    def test_zero_median_prob(self):
        self.mech.set_sensitivity(1).set_epsilon_delta(1, 0.1)
        vals = []

        for i in range(10000):
            vals.append(self.mech.randomise(0.5))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.5, delta=0.1)

    def test_neighbors_prob(self):
        runs = 10000
        delta = 0.1
        self.mech.set_sensitivity(1).set_epsilon_delta(1, delta)
        count = [0, 0]

        for i in range(runs):
            val0 = self.mech.randomise(0)
            if val0 <= 1 - self.mech._noise_bound:
                count[0] += 1

            val1 = self.mech.randomise(1)
            if val1 >= self.mech._noise_bound:
                count[1] += 1

        self.assertAlmostEqual(count[0] / runs, delta, delta=delta/10)
        self.assertAlmostEqual(count[1] / runs, delta, delta=delta/10)

    def test_within_bounds(self):
        self.mech.set_sensitivity(1).set_epsilon_delta(1, 0.1)
        vals = []

        for i in range(1000):
            vals.append(self.mech.randomise(0))

        vals = np.array(vals)

        self.assertTrue(np.all(vals >= -self.mech._noise_bound))
        self.assertTrue(np.all(vals <= self.mech._noise_bound))

    def test_repr(self):
        repr_ = repr(self.mech.set_epsilon_delta(1, 0.1).set_sensitivity(1))
        self.assertIn(".LaplaceBoundedNoise(", repr_)

    def test_bias(self):
        self.mech.set_epsilon_delta(1, 0.1).set_sensitivity(1)
        self.assertEqual(self.mech.get_bias(0), 0.0)

    def test_variance(self):
        self.mech.set_epsilon_delta(1, 0.1).set_sensitivity(1)
        self.assertRaises(NotImplementedError, self.mech.get_variance, 0)
