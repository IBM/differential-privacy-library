from unittest import TestCase

from diffprivlib.mechanisms import TruncationAndFoldingMixin, DPMechanism


class TestTruncationAndFoldingMixin(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(TruncationAndFoldingMixin)

    def test_lone_instantiation(self):
        with self.assertRaises(TypeError):
            TruncationAndFoldingMixin()

    def test_dummy_instantiation(self):
        class TestClass(DPMechanism, TruncationAndFoldingMixin):
            def randomise(self, value):
                return 0

        mech = TestClass()
        self.assertEqual(mech.randomise(0), 0)
