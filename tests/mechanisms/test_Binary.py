import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Binary
from diffprivlib.utils import global_seed

mech = Binary()
global_seed(3141592653)


class TestBinary(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Binary, DPMechanism))

    def test_no_params(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.randomise("1")

    def test_no_labels(self):
        _mech = mech.copy().set_epsilon(1)
        with self.assertRaises(ValueError):
            _mech.randomise("1")

    def test_no_epsilon(self):
        _mech = mech.copy().set_labels("0", "1")
        with self.assertRaises(ValueError):
            _mech.randomise("1")

    def test_inf_epsilon(self):
        _mech = mech.copy().set_labels("0", "1").set_epsilon(float("inf"))

        for i in range(1000):
            self.assertEqual(_mech.randomise("1"), "1")
            self.assertEqual(_mech.randomise("0"), "0")

    def test_complex_epsilon(self):
        _mech = mech.copy().set_labels("0", "1")
        with self.assertRaises(TypeError):
            _mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        _mech = mech.copy().set_labels("0", "1")
        with self.assertRaises(TypeError):
            _mech.set_epsilon("Two")

    def test_non_string_labels(self):
        with self.assertRaises(TypeError):
            mech.copy().set_epsilon(1).set_labels(0, 1)

    def test_non_string_input(self):
        _mech = mech.copy().set_epsilon(1).set_labels("0", "1")
        with self.assertRaises(TypeError):
            _mech.randomise(0)

    def test_empty_label(self):
        with self.assertRaises(ValueError):
            mech.copy().set_epsilon(1).set_labels("0", "")

    def test_same_labels(self):
        with self.assertRaises(ValueError):
            mech.copy().set_epsilon(1).set_labels("0", "0")

    def test_distrib(self):
        epsilon = np.log(2)
        runs = 20000
        _mech = mech.copy().set_epsilon(epsilon).set_labels("0", "1")
        count = [0, 0]

        for i in range(runs):
            val = _mech.randomise("0")
            count[int(val)] += 1

        # print("%d / %d = %f" % (count[0], count[1], count[0] / count[1]))
        self.assertAlmostEqual(count[0] / count[1], np.exp(epsilon), delta=0.1)
