from unittest import TestCase

from diffprivlib.mechanisms import TruncationAndFoldingMachine, DPMechanism


class TestTruncationAndFoldingMachine(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(TruncationAndFoldingMachine)

    def test_lone_instantiation(self):
        with self.assertRaises(TypeError):
            TruncationAndFoldingMachine()

    def test_dummy_instantiation(self):
        class TestClass(DPMechanism, TruncationAndFoldingMachine):
            def randomise(self, value):
                return 0

        mech = TestClass()
        self.assertEqual(mech.randomise(0), 0)
