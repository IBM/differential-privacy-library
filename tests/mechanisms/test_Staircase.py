import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Staircase
from diffprivlib.utils import global_seed

global_seed(3141592653)
mech = Staircase()


class TestStaircase(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Staircase, DPMechanism))

    def test_no_params(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_no_sensitivity(self):
        _mech = mech.copy().set_epsilon(1).set_gamma(0.5)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_no_epsilon(self):
        _mech = mech.copy().set_sensitivity(1).set_gamma(0.5)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_neg_epsilon(self):
        _mech = mech.copy().set_sensitivity(1).set_gamma(0.5)
        with self.assertRaises(ValueError):
            _mech.set_epsilon(-1)

    def test_complex_epsilon(self):
        _mech = mech.copy()
        with self.assertRaises(TypeError):
            _mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        _mech = mech.copy()
        with self.assertRaises(TypeError):
            _mech.set_epsilon("Two")

    def test_non_zero_delta(self):
        _mech = mech.copy().set_sensitivity(1).set_gamma(0.5)
        with self.assertRaises(ValueError):
            _mech.set_epsilon_delta(1, 0.5)

    def test_non_numeric(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1).set_gamma(0.5)
        with self.assertRaises(TypeError):
            _mech.randomise("Hello")

    def test_zero_median(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon(1).set_gamma(0.5)
        vals = []

        for i in range(10000):
            vals.append(_mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0, delta=0.1)

    def test_neighbors(self):
        epsilon = 1
        runs = 10000
        _mech = mech.copy().set_sensitivity(1).set_epsilon(epsilon).set_gamma(0.5)
        count = [0, 0]

        for i in range(runs):
            val0 = _mech.randomise(0)
            if val0 <= 0:
                count[0] += 1

            val1 = _mech.randomise(1)
            if val1 <= 0:
                count[1] += 1

        # print("0: %d; 1: %d" % (count[0], count[1]))
        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)

    def test_geometric(self):
        epsilon = 1
        count = [0, 0, 0, 0]
        runs = 200000
        b = np.exp(- epsilon)

        for i in range(runs):
            g = np.random.geometric(1 - b) - 1

            if g <= 3:
                count[int(g)] += 1

        self.assertAlmostEqual(count[0] / runs, (1 - b) * b ** 0, delta=0.005)
        self.assertAlmostEqual(count[1] / runs, (1 - b) * b ** 1, delta=0.005)
        self.assertAlmostEqual(count[2] / runs, (1 - b) * b ** 2, delta=0.005)
        self.assertAlmostEqual(count[3] / runs, (1 - b) * b ** 3, delta=0.005)
