import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Uniform

mech = Uniform()


class TestUniform(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Uniform, DPMechanism))

    def test_no_params(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_no_delta(self):
        _mech = mech.copy().set_sensitivity(1)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_large_delta(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.set_epsilon_delta(0, 0.6)

    def test_zero_delta(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.set_epsilon_delta(0, 0)

    def test_nonzero_epsilon(self):
        _mech = mech.copy()
        with self.assertRaises(ValueError):
            _mech.set_epsilon_delta(1, 0.2)

    def test_complex_delta(self):
        _mech = mech.copy()
        with self.assertRaises(TypeError):
            _mech.set_epsilon_delta(0, 0.5+2j)

    def test_string_delta(self):
        _mech = mech.copy()
        with self.assertRaises(TypeError):
            _mech.set_epsilon_delta(0, "Half")

    def test_no_sensitivity(self):
        _mech = mech.copy().set_epsilon_delta(0, 0.2)
        with self.assertRaises(ValueError):
            _mech.randomise(1)

    def test_non_numeric(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon_delta(0, 0.2)
        with self.assertRaises(TypeError):
            _mech.randomise("Hello")

    def test_zero_median(self):
        _mech = mech.copy().set_sensitivity(1).set_epsilon_delta(0, 0.2)
        vals = []

        for i in range(10000):
            vals.append(_mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)
