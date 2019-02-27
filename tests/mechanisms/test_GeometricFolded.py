import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import GeometricFolded

mech = GeometricFolded()


class TestGeometricFolded(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(GeometricFolded, DPMechanism))

    def test_no_params(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_no_sensitivity(self):
        _mech = mech.copy().set_epsilon(1).set_bounds(0, 10)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_non_integer_sensitivity(self):
        _mech = mech.copy().set_epsilon(1).set_bounds(0, 10)
        with self.assertRaises(TypeError):
            _mech.set_sensitivity(0.5)

    def test_no_epsilon(self):
        _mech = mech.copy().set_sensitivity(1).set_bounds(0, 10)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_non_zero_delta(self):
        _mech = mech.copy().set_sensitivity(1).set_bounds(0, 10)
        with self.assertRaises(ValueError):
            mech.copy().set_epsilon_delta(1, 0.5)

    def test_neg_epsilon(self):
        _mech = mech.copy().set_sensitivity(1).set_bounds(0, 10)
        with self.assertRaises(ValueError):
            _mech.set_epsilon(-1)

    def test_no_bounds(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_half_integer_bounds(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1).set_bounds(0, 1.5)
        val = _mech.randomise(0)
        self.assertTrue(isinstance(val, int))

    def test_non_half_integer_bounds(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1)
        with self.assertRaises(ValueError):
            _mech.set_bounds(1, 2.2)

    def test_non_numeric(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1).set_bounds(0, 10)
        with self.assertRaises(TypeError):
            _mech.randomise("Hello")

    def test_non_integer(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1).set_bounds(0, 10)
        with self.assertRaises(TypeError):
            _mech.randomise(1.0)

    def test_zero_median(self):
        _mech = mech.copy().set_sensitivity(1).set_bounds(0, 4).set_epsilon(1)
        vals = []

        for i in range(10000):
            vals.append(_mech.randomise(2))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 2.0, delta=0.1)

    def test_neighbors(self):
        epsilon = 1
        runs = 10000
        _mech = mech.copy().set_sensitivity(1).set_epsilon(epsilon).set_bounds(0, 4)
        count = [0, 0]

        for i in range(runs):
            val0 = _mech.randomise(1)
            if val0 <= 1:
                count[0] += 1

            val1 = _mech.randomise(2)
            if val1 <= 1:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)
