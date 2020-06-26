import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import GaussianDiscrete
from diffprivlib.utils import global_seed


class TestGaussianDiscrete(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = GaussianDiscrete()

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(GaussianDiscrete, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_no_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_no_delta(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon(0.5)

    def test_no_sensitivity(self):
        self.mech.set_epsilon_delta(1.5, 0.1)
        self.assertEqual(1, self.mech._sensitivity)
        self.assertIsNotNone(self.mech.randomise(0))

    def test_neg_sensitivity(self):
        self.mech.set_epsilon_delta(1.5, 0.1)
        with self.assertRaises(ValueError):
            self.mech.set_sensitivity(-1)

    def test_non_int_sensitivity(self):
        self.mech.set_epsilon_delta(1.5, 0.1)
        with self.assertRaises(TypeError):
            self.mech.set_sensitivity(1.5)

    def test_zero_sensitivity(self):
        self.mech.set_epsilon_delta(1, 0.1).set_sensitivity(0)

        for i in range(1000):
            self.assertAlmostEqual(self.mech.randomise(1), 1)

    def test_large_epsilon(self):
        self.mech.set_epsilon_delta(1.5, 0.1)
        self.assertIsNotNone(self.mech.randomise(0))

    def test_inf_epsilon(self):
        self.mech.set_epsilon_delta(float("inf"), 0.1).set_sensitivity(1)

        for i in range(1000):
            self.assertAlmostEqual(self.mech.randomise(1), 1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon_delta(1+2j, 0.1)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon_delta("Two", 0.1)

    def test_non_numeric(self):
        self.mech.set_epsilon_delta(0.5, 0.1)
        with self.assertRaises(TypeError):
            self.mech.randomise("Hello")

    def test_non_int(self):
        self.mech.set_epsilon_delta(0.5, 0.1)
        with self.assertRaises(TypeError):
            self.mech.randomise(1.1)

    def test_simple(self):
        self.mech.set_epsilon_delta(2, 0.1)

        self.assertIsNotNone(self.mech.randomise(3))

    def test_zero_median_prob(self):
        self.mech.set_epsilon_delta(0.5, 0.1)
        vals = []

        for i in range(10000):
            vals.append(self.mech.randomise(1))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 1, delta=0.1)

    def test_zero_median_sens_prob(self):
        self.mech.set_epsilon_delta(0.5, 0.1).set_sensitivity(4)
        vals = []

        for i in range(10000):
            vals.append(self.mech.randomise(1))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 1, delta=0.1)

    def test_neighbors_prob(self):
        epsilon = 1
        runs = 10000
        self.mech.set_epsilon_delta(epsilon, 0.1)
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

    def test_sanity_scale(self):
        self.mech.set_epsilon_delta(1, 0.5).check_inputs(1)
        scale1 = self.mech._scale

        self.mech.set_epsilon_delta(2, 0.75).check_inputs(1)
        scale2 = self.mech._scale

        self.assertGreater(scale1, scale2)

        self.mech.set_epsilon_delta(3, 4.00902e-10).check_inputs(1)
        scale3 = self.mech._scale

        self.assertAlmostEqual(2, scale3, places=5)

    def test_bias(self):
        self.assertEqual(0, self.mech.get_bias(0))

    def test_variance(self):
        self.assertRaises(NotImplementedError, self.mech.get_variance, 0)

    def test_repr(self):
        repr_ = repr(self.mech.set_epsilon_delta(1, 0.5))
        self.assertIn(".GaussianDiscrete(", repr_)

    def test_bernoulli_exp_prob(self):
        vals = []
        runs = 30000

        for i in range(runs):
            vals.append(self.mech._bernoulli_exp(-np.log(0.5)))

        self.assertAlmostEqual(sum(vals) / runs, 0.5, places=2)

        vals = []
        for i in range(runs):
            vals.append(self.mech._bernoulli_exp(-np.log(0.1)))

        self.assertAlmostEqual(sum(vals) / runs, 0.1, places=2)
