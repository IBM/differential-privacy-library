from unittest import TestCase

from numpy.random import random

from diffprivlib.mechanisms import DPMachine, DPMechanism


class TestDPMechanism(TestCase):
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
