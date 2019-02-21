from unittest import TestCase

from diffprivlib import DPMachine
from diffprivlib.mechanisms import DPMechanism


class TestDPMechanism(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(DPMechanism)

    def test_parent_class(self):
        self.assertTrue(issubclass(DPMechanism, DPMachine))

    def test_instantiation(self):
        with self.assertRaises(TypeError):
            DPMechanism()
