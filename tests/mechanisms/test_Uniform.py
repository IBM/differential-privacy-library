import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Uniform
from diffprivlib.utils import global_seed


class TestUniform(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Uniform()

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Uniform, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_no_delta(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_large_delta(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(0, 0.6)

    def test_zero_delta(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(0, 0)

    def test_nonzero_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(1, 0.2)

    def test_complex_delta(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon_delta(0, 0.5+2j)

    def test_string_delta(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon_delta(0, "Half")

    def test_no_sensitivity(self):
        self.mech.set_epsilon_delta(0, 0.2)
        with self.assertRaises(ValueError):
            self.mech.randomise(1)

    def test_wrong_sensitivity(self):
        self.mech.set_epsilon_delta(0, 0.2)

        with self.assertRaises(ValueError):
            self.mech.set_sensitivity(-1)

        with self.assertRaises(TypeError):
            self.mech.set_sensitivity("1")

    def test_zero_sensitivity(self):
        self.mech.set_epsilon_delta(0, 0.2).set_sensitivity(0)

        for i in range(1000):
            self.assertAlmostEqual(self.mech.randomise(1), 1)

    def test_non_numeric(self):
        self.mech.set_sensitivity(1).set_epsilon_delta(0, 0.2)
        with self.assertRaises(TypeError):
            self.mech.randomise("Hello")

    def test_zero_median_prob(self):
        self.mech.set_sensitivity(1).set_epsilon_delta(0, 0.2)
        vals = []

        for i in range(10000):
            vals.append(self.mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_repr(self):
        repr_ = repr(self.mech.set_epsilon_delta(0, 0.1).set_sensitivity(1))
        self.assertIn(".Uniform(", repr_)

    def test_bias(self):
        self.assertEqual(0.0, self.mech.get_bias(0))

    def test_variance(self):
        self.mech.set_epsilon_delta(0, 0.1).set_sensitivity(1)
        self.assertGreater(self.mech.get_variance(0), 0.0)
