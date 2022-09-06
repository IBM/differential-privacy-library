import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import LaplaceBoundedNoise


class TestLaplaceBoundedNoise(TestCase):
    def setup_method(self, method):
        self.mech = LaplaceBoundedNoise

    def teardown_method(self, method):
        del self.mech

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(LaplaceBoundedNoise, DPMechanism))

    def test_zero_sensitivity(self):
        mech = self.mech(epsilon=1, delta=0.1, sensitivity=0)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=-1, delta=0.1, sensitivity=1)

    def test_inf_epsilon(self):
        mech = self.mech(epsilon=float("inf"), delta=0.1, sensitivity=1)

        for i in range(1000):
            self.assertAlmostEqual(mech.randomise(1), 1)

    def test_zero_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=0, delta=0.1, sensitivity=1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon=1 + 2j, delta=0.1, sensitivity=1)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech(epsilon="Two", delta=0.1, sensitivity=1)

    def test_zero_delta(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, delta=0, sensitivity=1)

    def test_large_delta(self):
        with self.assertRaises(ValueError):
            self.mech(epsilon=1, delta=0.6, sensitivity=1)

    def test_non_numeric(self):
        mech = self.mech(epsilon=1, delta=0.1, sensitivity=1)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_non_nan(self):
        mech = self.mech(epsilon=1, delta=0.1, sensitivity=1)
        self.assertTrue(np.isnan(mech.randomise(np.nan)))

    def test_zero_median_prob(self):
        mech = self.mech(epsilon=1, delta=0.1, sensitivity=1, random_state=0)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_neighbors_prob(self):
        runs = 1000
        delta = 0.1
        mech = self.mech(epsilon=1, delta=delta, sensitivity=1, random_state=2)

        count0 = (np.array([mech.randomise(0) for _ in range(runs)]) <= 1 - mech._noise_bound).sum()
        count1 = (np.array([mech.randomise(1) for _ in range(runs)]) >= mech._noise_bound).sum()

        self.assertAlmostEqual(count0 / runs, delta, delta=delta / 10)
        self.assertAlmostEqual(count1 / runs, delta, delta=delta / 10)

    def test_within_bounds(self):
        mech = self.mech(epsilon=1, delta=0.1, sensitivity=1)
        vals = []

        for i in range(1000):
            vals.append(mech.randomise(0))

        vals = np.array(vals)

        self.assertTrue(np.all(vals >= -mech._noise_bound))
        self.assertTrue(np.all(vals <= mech._noise_bound))

    def test_random_state(self):
        mech1 = self.mech(epsilon=1, delta=1e-3, sensitivity=1, random_state=42)
        mech2 = self.mech(epsilon=1, delta=1e-3, sensitivity=1, random_state=42)
        self.assertEqual([mech1.randomise(0) for _ in range(100)], [mech2.randomise(0) for _ in range(100)])

        self.assertNotEqual([mech1.randomise(0)] * 100, [mech1.randomise(0) for _ in range(100)])

        rng = np.random.RandomState(0)
        mech1 = self.mech(epsilon=1, delta=1e-3, sensitivity=1, random_state=rng)
        mech2 = self.mech(epsilon=1, delta=1e-3, sensitivity=1, random_state=rng)
        self.assertNotEqual([mech1.randomise(0) for _ in range(100)], [mech2.randomise(0) for _ in range(100)])

    def test_repr(self):
        repr_ = repr(self.mech(epsilon=1, delta=0.1, sensitivity=1))
        self.assertIn(".LaplaceBoundedNoise(", repr_)

    def test_bias(self):
        mech = self.mech(epsilon=1, delta=0.1, sensitivity=1)
        self.assertEqual(mech.bias(0), 0.0)

    def test_variance(self):
        mech = self.mech(epsilon=1, delta=0.1, sensitivity=1)
        self.assertRaises(NotImplementedError, mech.variance, 0)
