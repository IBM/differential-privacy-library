from unittest import TestCase
import numpy as np

from diffprivlib.mechanisms import Laplace
from diffprivlib.utils import global_seed


class TestLaplace(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Laplace()

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Laplace, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_no_sensitivity(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_no_epsilon(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_neg_epsilon(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon(-1)

    def test_inf_epsilon(self):
        self.mech.set_sensitivity(1).set_epsilon(float("inf"))

        for i in range(1000):
            self.assertAlmostEqual(self.mech.randomise(1), 1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon("Two")

    def test_zero_epsilon_with_delta(self):
        self.mech.set_sensitivity(1).set_epsilon_delta(0, 0.5)
        self.assertIsNotNone(self.mech.randomise(1))

    def test_epsilon_delta(self):
        self.mech.set_sensitivity(1).set_epsilon_delta(1, 0.01)
        self.assertIsNotNone(self.mech.randomise(1))

    def test_non_numeric(self):
        self.mech.set_sensitivity(1).set_epsilon(1)
        with self.assertRaises(TypeError):
            self.mech.randomise("Hello")

    def test_zero_median_prob(self):
        self.mech.set_sensitivity(1).set_epsilon(1)
        vals = []

        for i in range(10000):
            vals.append(self.mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_neighbours_prob(self):
        epsilon = 1
        runs = 10000
        self.mech.set_sensitivity(1).set_epsilon(epsilon)
        count = [0, 0]

        for i in range(runs):
            val0 = self.mech.randomise(0)
            if val0 <= 0:
                count[0] += 1

            val1 = self.mech.randomise(1)
            if val1 <= 0:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)
