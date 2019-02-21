from unittest import TestCase
import numpy as np

from diffprivlib.mechanisms import Laplace


class TestLaplace(TestCase):
    def test_not_none(self):
        mech = Laplace()
        self.assertIsNotNone(mech)

    def test_no_params(self):
        mech = Laplace()
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_no_sensitivity(self):
        mech = Laplace().set_epsilon(1)
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_no_epsilon(self):
        mech = Laplace().set_sensitivity(1)
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_non_numeric(self):
        mech = Laplace().set_sensitivity(1).set_epsilon(1)
        with self.assertRaises(TypeError):
            mech.randomise("Hello")

    def test_zero_median(self):
        mech = Laplace().set_sensitivity(1).set_epsilon(1)
        vals = []

        for i in range(10000):
            vals.append(mech.randomise(0))

        median = float(np.median(vals))
        self.assertAlmostEqual(np.abs(median), 0.0, delta=0.1)

    def test_neighbours(self):
        epsilon = 1
        runs = 10000
        mech = Laplace().set_sensitivity(1).set_epsilon(epsilon)
        count = [0, 0]

        for i in range(runs):
            val0 = mech.randomise(0)
            if val0 <= 0:
                count[0] += 1

            val1 = mech.randomise(1)
            if val1 <= 0:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.1)
