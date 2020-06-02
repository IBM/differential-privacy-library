from unittest import TestCase

from numpy.random import random

from diffprivlib.mechanisms import DPMachine, DPMechanism


class TestDPMechanism(TestCase):
    def setup_method(self, method):
        class TestMech(DPMechanism):
            def randomise(self, value):
                return value

        self.mech = TestMech()

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(DPMechanism)

    def test_parent_class(self):
        self.assertTrue(issubclass(DPMechanism, DPMachine))

    def test_instantiation(self):
        with self.assertRaises(TypeError):
            DPMechanism()

    def test_neg_epsilon(self):
        class BaseDPMechanism(DPMechanism):
            def randomise(self, value):
                return random()

        mech = BaseDPMechanism()
        with self.assertRaises(ValueError):
            mech.set_epsilon(-1)

    def test_copy(self):
        self.assertIsInstance(self.mech.copy(), DPMechanism)
        self.assertIsInstance(self.mech.deepcopy(), DPMechanism)

    def test_set_epsilon_delta(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(0, -1)

        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(0, 2)

        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(0, 0)

        mech1 = self.mech.deepcopy()
        self.assertIsNotNone(mech1.set_epsilon(1))

        mech2 = self.mech.deepcopy()
        self.assertIsNotNone(mech2.set_epsilon_delta(1, 0.5))

    def test_mse(self):
        self.assertRaises(NotImplementedError, self.mech.get_mse, 1)

        class TestMSE(self.mech.__class__):
            def get_bias(self, value):
                return -1

            def get_variance(self, value):
                return 1

        mse_mech = TestMSE()

        self.assertEqual(mse_mech.get_mse(1), 2)
