import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import LaplaceBoundedNoise
from diffprivlib.utils import global_seed

global_seed(3141592653)
mech = LaplaceBoundedNoise()


class TestLaplaceBoundedDomain(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(LaplaceBoundedNoise, DPMechanism))

    def test_no_params(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_no_sensitivity(self):
        _mech = mech.copy().set_epsilon_delta(1, 0.1)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_no_epsilon(self):
        _mech = mech.copy().set_sensitivity(1)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_neg_epsilon(self):
        _mech = mech.copy().set_sensitivity(1)
        with self.assertRaises(ValueError):
            _mech.set_epsilon_delta(-1, 0.1)

    def test_inf_epsilon(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon_delta(float("inf"), 0.1)

        for i in range(1000):
            self.assertEqual(_mech.randomise(1), 1)

    def test_zero_epsilon(self):
        _mech = mech.copy().set_sensitivity(1)
        with self.assertRaises(ValueError):
            _mech.set_epsilon_delta(0, 0.1)

    def test_complex_epsilon(self):
        _mech = mech.copy()
        with self.assertRaises(TypeError):
            _mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        _mech = mech.copy()
        with self.assertRaises(TypeError):
            _mech.set_epsilon("Two")

    def test_no_delta(self):
        _mech = mech.copy().set_sensitivity(1)
        with self.assertRaises(ValueError):
            _mech.set_epsilon(1)

    def test_large_delta(self):
        _mech = mech.copy().set_sensitivity(1)
        with self.assertRaises(ValueError):
            _mech.set_epsilon_delta(1, 0.6)

    def test_non_numeric(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon_delta(1, 0.1)
        with self.assertRaises(TypeError):
            _mech.randomise("Hello")

    def test_zero_median(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon_delta(1, 0.1)
        vals = []

        for i in range(10000):
            vals.append(_mech.randomise(0.5))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.5, delta=0.1)

    def test_neighbors(self):
        runs = 10000
        delta = 0.1
        _mech = mech.copy().set_sensitivity(1).set_epsilon_delta(1, delta)
        count = [0, 0]

        for i in range(runs):
            val0 = _mech.randomise(0)
            if val0 <= 1 - _mech._noise_bound:
                count[0] += 1

            val1 = _mech.randomise(1)
            if val1 >= _mech._noise_bound:
                count[1] += 1

        self.assertAlmostEqual(count[0] / runs, delta, delta=delta/10)
        self.assertAlmostEqual(count[1] / runs, delta, delta=delta/10)

    def test_within_bounds(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon_delta(1, 0.1)
        vals = []

        for i in range(1000):
            vals.append(_mech.randomise(0))

        vals = np.array(vals)

        self.assertTrue(np.all(vals >= -_mech._noise_bound))
        self.assertTrue(np.all(vals <= _mech._noise_bound))
