import warnings

import numpy as np
from unittest import TestCase

import pytest

from diffprivlib.mechanisms import Wishart
from diffprivlib.utils import global_seed


class TestWishart(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Wishart

        # with warnings.catch_warnings(record=True) as w:
        #     warnings.filterwarnings('always', '', DeprecationWarning)
        #     self.mech = Wishart()
        #     self.assertIs(w[0].category, DeprecationWarning)

        self.random_array = np.random.randn(5,5)

    def teardown_method(self, method):
        del self.mech

    @staticmethod
    def generate_data(d=5):
        return np.random.randn(d, d)

    def test_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', DeprecationWarning)
            self.mech(epsilon=1, sensitivity=1)
            self.assertIs(w[0].category, DeprecationWarning)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Wishart, DPMechanism))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, sensitivity=1)

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), sensitivity=1)

        for i in range(100):
            data = self.generate_data()
            noisy_data = mech.randomise(data)
            self.assertTrue(np.allclose(data, noisy_data))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_wrong_sensitivity(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1, sensitivity="1")

        with self.assertRaises(ValueError):
            self.mech(epsilon=1, sensitivity=-1)

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, sensitivity=0)

        for i in range(100):
            data = self.generate_data()
            noisy_data = mech.randomise(data)
            self.assertTrue(np.allclose(data, noisy_data))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_numeric_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise(1)

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_string_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise("1")

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_list_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise([1, 2, 3])

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_string_array_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(TypeError):
            mech.randomise(np.array([["1", "2"], ["3", "4"]]))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_scalar_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(ValueError):
            mech.randomise(np.array([1]))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_scalar_array_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        self.assertIsNotNone(mech.randomise(np.array([[1]])))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_vector_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(ValueError):
            mech.randomise(np.array([1, 2, 3]))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_non_square_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(ValueError):
            mech.randomise(np.ones((3, 4)))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_3D_input(self):
        mech = self.mech(epsilon=1, sensitivity=1)

        with self.assertRaises(ValueError):
            mech.randomise(np.ones((3, 3, 3)))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_different_result(self):
        mech = self.mech(epsilon=1, sensitivity=1)
        data = self.generate_data()
        noisy_data = mech.randomise(data)

        for i in range(10):
            old_noisy_data = noisy_data
            noisy_data = mech.randomise(self.generate_data())

            self.assertFalse(np.allclose(noisy_data, old_noisy_data))

    @pytest.mark.filterwarnings("ignore:The Wishart mechanism has been shown")
    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, sensitivity=1))
        self.assertIn(".Wishart(", repr_)
