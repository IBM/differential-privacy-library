import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Vector
from diffprivlib.utils import global_seed


class TestVector(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Vector()

    def teardown_method(self, method):
        del self.mech

    @staticmethod
    def func(x):
        return np.sum(x ** 2)

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Vector, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise(self.func)

    def test_no_epsilon(self):
        self.mech.set_dimension(3).set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.randomise(self.func)

    def test_neg_epsilon(self):
        self.mech.set_dimension(3).set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon(-1)

    def test_inf_epsilon(self):
        self.mech.set_dimension(3).set_sensitivity(1).set_epsilon(float("inf"))

        for i in range(100):
            noisy_func = self.mech.randomise(self.func)
            self.assertAlmostEqual(noisy_func(np.zeros(3)), 0)
            self.assertAlmostEqual(noisy_func(np.ones(3)), 3)

    def test_nonzero_delta(self):
        self.mech.set_dimension(3).set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(1, 0.1)

    def test_wrong_sensitivity(self):
        self.mech.set_dimension(3).set_epsilon(1)
        with self.assertRaises(TypeError):
            self.mech.set_sensitivity("1")

        with self.assertRaises(TypeError):
            self.mech.set_sensitivity(1, "1")

        with self.assertRaises(ValueError):
            self.mech.set_sensitivity(-1)

        with self.assertRaises(ValueError):
            self.mech.set_sensitivity(1, -1)

    def test_no_sensitivity(self):
        self.mech.set_dimension(3).set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.randomise(self.func)

    def test_zero_data_sensitivity(self):
        self.mech.set_dimension(3).set_sensitivity(1, 0).set_epsilon(1)

        for i in range(100):
            noisy_func = self.mech.randomise(self.func)
            self.assertAlmostEqual(noisy_func(np.zeros(3)), 0)
            self.assertAlmostEqual(noisy_func(np.ones(3)), 3)

    def test_wrong_alpha(self):
        self.mech.set_dimension(3).set_epsilon(1)

        with self.assertRaises(TypeError):
            self.mech.set_alpha("1")

        with self.assertRaises(ValueError):
            self.mech.set_alpha(-1)

    def test_wrong_dimension(self):
        self.mech.set_epsilon(1)

        with self.assertRaises(TypeError):
            self.mech.set_dimension(1.2)

        with self.assertRaises(ValueError):
            self.mech.set_dimension(0)

    def test_no_dimension(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(ValueError):
            self.mech.randomise(self.func)

    def test_numeric_input(self):
        self.mech.set_dimension(3).set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(TypeError):
            self.mech.randomise(1)

    def test_string_input(self):
        self.mech.set_dimension(3).set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(TypeError):
            self.mech.randomise("1")

    def test_sets_once(self):
        self.mech.set_dimension(3).set_epsilon(1).set_sensitivity(1)
        noisy_func = self.mech.randomise(self.func)
        answer = noisy_func(np.ones(3))

        for i in range(10):
            self.assertEqual(noisy_func(np.ones(3)), answer)

    def test_different_result(self):
        self.mech.set_dimension(3).set_epsilon(1).set_sensitivity(1)
        noisy_func = self.mech.randomise(self.func)

        for i in range(10):
            old_noisy_func = noisy_func
            noisy_func = self.mech.randomise(self.func)

            self.assertNotAlmostEqual(noisy_func(np.ones(3)), 3)
            self.assertNotAlmostEqual(noisy_func(np.ones(3)), old_noisy_func(np.ones(3)))
            # print(noisy_func(np.ones(3)))

    def test_repr(self):
        repr_ = repr(self.mech.set_epsilon(1).set_dimension(4).set_sensitivity(1))
        self.assertIn(".Vector(", repr_)
