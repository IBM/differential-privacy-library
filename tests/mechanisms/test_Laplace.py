from unittest import TestCase
import numpy as np

from diffprivlib.mechanisms import Laplace
from diffprivlib.utils import global_seed

global_seed(3141592653)
mech = Laplace()


class TestLaplace(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Laplace, DPMechanism))

    def test_no_params(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_no_sensitivity(self):
        _mech = mech.copy().set_epsilon(1)
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_no_epsilon(self):
        mech = Laplace().set_sensitivity(1)
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_neg_epsilon(self):
        _mech = mech.copy().set_sensitivity(1)
        with self.assertRaises(ValueError):
            _mech.set_epsilon(-1)

    def test_inf_epsilon(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(float("inf"))

        for i in range(1000):
            self.assertAlmostEqual(_mech.randomise(1), 1)

    def test_complex_epsilon(self):
        _mech = mech.copy()
        with self.assertRaises(TypeError):
            _mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        _mech = mech.copy()
        with self.assertRaises(TypeError):
            _mech.set_epsilon("Two")

    def test_zero_epsilon_with_delta(self):
        mech = Laplace().set_sensitivity(1).set_epsilon_delta(0, 0.5)
        self.assertIsNotNone(mech.randomise(1))

    def test_epsilon_delta(self):
        mech = Laplace().set_sensitivity(1).set_epsilon_delta(1, 0.01)
        self.assertIsNotNone(mech.randomise(1))

    def test_non_numeric(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1)
        with self.assertRaises(TypeError):
            _mech.randomise("Hello")

    def test_zero_median(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1)
        vals = []

        for i in range(10000):
            vals.append(_mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_neighbours(self):
        epsilon = 1
        runs = 10000
        _mech = mech.copy().set_sensitivity(1).set_epsilon(epsilon)
        count = [0, 0]

        for i in range(runs):
            val0 = _mech.randomise(0)
            if val0 <= 0:
                count[0] += 1

            val1 = _mech.randomise(1)
            if val1 <= 0:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)
