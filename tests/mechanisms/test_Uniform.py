import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Uniform


class TestUniform(TestCase):
    def setup_method(self, method):
        self.mech = Uniform

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Uniform, DPMechanism))

    def test_large_delta(self):
        with self.assertRaises(ValueError):
            self.mech(delta=0.6, sensitivity=1)

    def test_zero_delta(self):
        with self.assertRaises(ValueError):
            self.mech(delta=0, sensitivity=1)

    def test_nonzero_epsilon(self):
        mech = self.mech(delta=0.1, sensitivity=1)
        mech.epsilon = 1
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_complex_delta(self):
        with self.assertRaises(TypeError):
            self.mech(delta=0.1 + 0.1j, sensitivity=1)

    def test_string_delta(self):
        with self.assertRaises(TypeError):
            self.mech(delta="0.1", sensitivity=1)

    def test_wrong_sensitivity(self):
        with self.assertRaises(ValueError):
            self.mech(delta=0.1, sensitivity=-1)

        with self.assertRaises(TypeError):
            self.mech(delta=0.1, sensitivity="1")

    def test_zero_sensitivity(self):
        mech = self.mech(delta=0.1, sensitivity=0)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_non_numeric(self):
        mech = self.mech(delta=0.1, sensitivity=1)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_zero_median_prob(self):
        mech = self.mech(delta=0.2, sensitivity=1, random_state=0)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_random_state(self):
        mech1 = self.mech(delta=0.2, sensitivity=1, random_state=42)
        mech2 = self.mech(delta=0.2, sensitivity=1, random_state=42)
        self.assertEqual([mech1.randomise(0) for _ in range(100)], [mech2.randomise(0) for _ in range(100)])

        self.assertNotEqual([mech1.randomise(0)] * 100, [mech1.randomise(0) for _ in range(100)])

        rng = np.random.RandomState(0)
        mech1 = self.mech(delta=0.2, sensitivity=1, random_state=rng)
        mech2 = self.mech(delta=0.2, sensitivity=1, random_state=rng)
        self.assertNotEqual([mech1.randomise(0) for _ in range(100)], [mech2.randomise(0) for _ in range(100)])

    def test_repr(self):
        repr_ = repr(self.mech(delta=0.1, sensitivity=1))
        self.assertIn(".Uniform(", repr_)

    def test_bias(self):
        self.assertEqual(0.0, self.mech(delta=0.1, sensitivity=1).bias(0))

    def test_variance(self):
        mech = self.mech(delta=0.1, sensitivity=1)
        self.assertGreater(mech.variance(0), 0.0)
