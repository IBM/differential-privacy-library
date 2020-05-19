import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Wishart
from diffprivlib.utils import global_seed


class TestWishart(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Wishart()
        self.random_array = np.random.randn(5,5)

    def teardown_method(self, method):
        del self.mech

    @staticmethod
    def generate_data(d=5):
        return np.random.randn(d, d)

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Wishart, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise(self.generate_data())

    def test_no_epsilon(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.randomise(self.generate_data())

    def test_neg_epsilon(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon(-1)

    def test_inf_epsilon(self):
        self.mech.set_sensitivity(1).set_epsilon(float("inf"))

        for i in range(100):
            data = self.generate_data()
            noisy_data = self.mech.randomise(data)
            self.assertTrue(np.allclose(data, noisy_data))

    def test_non_zero_delta(self):
        self.mech.set_sensitivity(1)
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(1, 0.5)

    def test_wrong_sensitivity(self):
        self.mech.set_epsilon(1)

        with self.assertRaises(TypeError):
            self.mech.set_sensitivity("1")

        with self.assertRaises(ValueError):
            self.mech.set_sensitivity(-1)

    def test_no_sensitivity(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(ValueError):
            self.mech.randomise(self.generate_data())

    def test_zero_sensitivity(self):
        self.mech.set_epsilon(1).set_sensitivity(0)

        for i in range(100):
            data = self.generate_data()
            noisy_data = self.mech.randomise(data)
            self.assertTrue(np.allclose(data, noisy_data))

    def test_numeric_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(TypeError):
            self.mech.randomise(1)

    def test_string_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(TypeError):
            self.mech.randomise("1")

    def test_list_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(TypeError):
            self.mech.randomise([1, 2, 3])

    def test_string_array_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(TypeError):
            self.mech.randomise(np.array([["1", "2"], ["3", "4"]]))

    def test_scalar_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(ValueError):
            self.mech.randomise(np.array([1]))

    def test_scalar_array_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        self.assertIsNotNone(self.mech.randomise(np.array([[1]])))

    def test_vector_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(ValueError):
            self.mech.randomise(np.array([1, 2, 3]))

    def test_non_square_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(ValueError):
            self.mech.randomise(np.ones((3, 4)))

    def test_3D_input(self):
        self.mech.set_epsilon(1).set_sensitivity(1)

        with self.assertRaises(ValueError):
            self.mech.randomise(np.ones((3, 3, 3)))

    def test_different_result(self):
        self.mech.set_epsilon(1).set_sensitivity(1)
        data = self.generate_data()
        noisy_data = self.mech.randomise(data)

        for i in range(10):
            old_noisy_data = noisy_data
            noisy_data = self.mech.randomise(self.generate_data())

            self.assertFalse(np.allclose(noisy_data, old_noisy_data))

    def test_repr(self):
        repr_ = repr(self.mech.set_epsilon(1).set_sensitivity(1))
        self.assertIn(".Wishart(", repr_)
