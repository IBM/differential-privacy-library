import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Vector
from diffprivlib.utils import global_seed

global_seed(3141592653)
mech = Vector()
func = lambda x: np.sum(x ** 2)


class TestVector(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Vector, DPMechanism))

    def test_no_params(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.randomise(func)

    def test_no_epsilon(self):
        _mech = mech.set_dimensions(3, 10)
        with self.assertRaises(ValueError):
            mech.randomise(func)

    def test_neg_epsilon(self):
        _mech = mech.copy().set_dimensions(3, 10)
        with self.assertRaises(ValueError):
            _mech.set_epsilon(-1)

    def test_inf_epsilon(self):
        _mech = mech.copy().set_dimensions(3, 10).set_epsilon(float("inf"))

        for i in range(100):
            noisy_func = _mech.randomise(func)
            self.assertAlmostEqual(noisy_func(np.zeros(3)), 0)
            self.assertAlmostEqual(noisy_func(np.ones(3)), 3)

    def test_numeric_input(self):
        _mech = mech.copy().set_dimensions(3, 10).set_epsilon(1)

        with self.assertRaises(TypeError):
            _mech.randomise(1)

    def test_string_input(self):
        _mech = mech.copy().set_dimensions(3, 10).set_epsilon(1)

        with self.assertRaises(TypeError):
            _mech.randomise("1")

    def test_different_result(self):
        _mech = mech.copy().set_dimensions(3, 10).set_epsilon(1)
        noisy_func = _mech.randomise(func)

        for i in range(10):
            old_noisy_func= noisy_func
            noisy_func = _mech.randomise(func)

            self.assertNotAlmostEqual(noisy_func(np.ones(3)), 3)
            self.assertNotAlmostEqual(noisy_func(np.ones(3)), old_noisy_func(np.ones(3)))
            # print(noisy_func(np.ones(3)))
