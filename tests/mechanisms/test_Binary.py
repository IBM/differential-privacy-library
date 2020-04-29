import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Binary
from diffprivlib.utils import global_seed


class TestBinary(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Binary()

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Binary, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise("1")

    def test_no_labels(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.randomise("1")

    def test_no_epsilon(self):
        self.mech.set_labels("0", "1")
        with self.assertRaises(ValueError):
            self.mech.randomise("1")

    def test_inf_epsilon(self):
        self.mech.set_labels("0", "1").set_epsilon(float("inf"))

        for i in range(1000):
            self.assertEqual(self.mech.randomise("1"), "1")
            self.assertEqual(self.mech.randomise("0"), "0")

    def test_complex_epsilon(self):
        self.mech.set_labels("0", "1")
        with self.assertRaises(TypeError):
            self.mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        self.mech.set_labels("0", "1")
        with self.assertRaises(TypeError):
            self.mech.set_epsilon("Two")

    def test_non_string_labels(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(TypeError):
            self.mech.set_labels(0, 1)

    def test_non_string_input(self):
        self.mech.set_epsilon(1).set_labels("0", "1")
        with self.assertRaises(TypeError):
            self.mech.randomise(0)

    def test_empty_label(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.set_labels("0", "")

    def test_same_labels(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.set_labels("0", "0")

    def test_randomise_without_labels(self):
        self.mech.set_epsilon(1).set_labels("1", "2")
        with self.assertRaises(ValueError):
            self.mech.randomise("0")

    def test_distrib_prob(self):
        epsilon = np.log(2)
        runs = 20000
        self.mech.set_epsilon(epsilon).set_labels("0", "1")
        count = [0, 0]

        for i in range(runs):
            val = self.mech.randomise("0")
            count[int(val)] += 1

        # print("%d / %d = %f" % (count[0], count[1], count[0] / count[1]))
        self.assertAlmostEqual(count[0] / count[1], np.exp(epsilon), delta=0.1)

    def test_repr(self):
        repr_ = repr(self.mech.set_epsilon(1).set_labels("0", "1"))
        self.assertIn(".Binary(", repr_)
