import abc
import sys
from unittest import TestCase

from diffprivlib import DPMachine

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class TestDPMachine(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(DPMachine)

    def test_class(self):
        self.assertTrue(issubclass(DPMachine, ABC))

    def test_instantiation(self):
        with self.assertRaises(TypeError):
            DPMachine()
