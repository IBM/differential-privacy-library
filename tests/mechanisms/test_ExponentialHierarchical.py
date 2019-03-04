import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import ExponentialHierarchical
from diffprivlib.utils import global_seed

global_seed(3141592653)
mech = ExponentialHierarchical()


class TestExponentialHierarchical(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(ExponentialHierarchical, DPMechanism))

    def test_no_params(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.randomise("A")

    def test_no_epsilon(self):
        _mech = mech.copy()
        with self.assertRaises(RuntimeError):
            _mech.set_hierarchy([["A", "B"], ["C"]])

    def test_neg_epsilon(self):
        _mech = mech.copy()
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
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.set_epsilon_delta(1, 0.5)

    def test_non_string_hierarchy(self):
        _mech = mech.copy().set_epsilon(1)
        with self.assertRaises(TypeError):
            _mech.set_hierarchy([["A", "B"], ["C", 2]])

    def test_uneven_hierarchy(self):
        _mech = mech.copy().set_epsilon(1)
        with self.assertRaises(ValueError):
            _mech.set_hierarchy(["A", ["B", "C"]])

    def test_non_string_input(self):
        _mech = mech.copy().set_epsilon(1).set_hierarchy([["A", "B"], ["C", "2"]])
        with self.assertRaises(TypeError):
            _mech.randomise(2)

    def test_outside_domain(self):
        _mech = mech.copy().set_epsilon(1).set_hierarchy([["A", "B"], ["C"]])
        with self.assertRaises(ValueError):
            _mech.randomise("D")

    def test_distrib(self):
        epsilon = np.log(2)
        runs = 20000
        _mech = mech.copy().set_epsilon(epsilon).set_hierarchy([["A", "B"], ["C"]])
        count = [0, 0, 0]

        for i in range(runs):
            val = _mech.randomise("A")
            if val == "A":
                count[0] += 1
            elif val == "B":
                count[1] += 1
            elif val == "C":
                count[2] += 1

        # print("A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertAlmostEqual(count[0] / runs, np.exp(epsilon) * count[2] / runs, delta=0.1)
        self.assertAlmostEqual(count[0] / count[1], count[1] / count[2], delta=0.1)
