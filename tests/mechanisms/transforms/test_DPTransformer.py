from numbers import Real
from unittest import TestCase

from diffprivlib.mechanisms import ExponentialHierarchical, Laplace, Gaussian
from diffprivlib.mechanisms.transforms import DPTransformer


class TestDPTransformer(TestCase):
    def test_not_none(self):
        mech = DPTransformer(ExponentialHierarchical())
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMachine
        self.assertTrue(issubclass(DPTransformer, DPMachine))

    def test_no_parent(self):
        with self.assertRaises(TypeError):
            DPTransformer()

    def test_bad_parent(self):
        with self.assertRaises(TypeError):
            DPTransformer(int)

    def test_empty_mechanism(self):
        mech = DPTransformer(ExponentialHierarchical())
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_nested(self):
        mech = DPTransformer(DPTransformer(DPTransformer(ExponentialHierarchical())))
        self.assertIsNotNone(mech)

    def test_epsilon_locally(self):
        mech = DPTransformer(Laplace().set_sensitivity(1))
        mech.set_epsilon(1)
        self.assertIsNotNone(mech)

    def test_epsilon_delta_locally(self):
        mech = DPTransformer(Gaussian().set_sensitivity(1))
        mech.set_epsilon_delta(0.5, 0.1)
        self.assertIsNotNone(mech)

    def test_laplace(self):
        mech = DPTransformer(Laplace().set_epsilon(1).set_sensitivity(1))
        self.assertIsInstance(mech.randomise(1), Real)
