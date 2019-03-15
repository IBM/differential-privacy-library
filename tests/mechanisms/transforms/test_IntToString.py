import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import ExponentialHierarchical
from diffprivlib.mechanisms.transforms import IntToString


class TestIntToString(TestCase):
    def test_not_none(self):
        mech = IntToString(ExponentialHierarchical())
        self.assertIsNotNone(mech)
        _mech = mech.copy()
        self.assertIsNotNone(_mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMachine
        self.assertTrue(issubclass(IntToString, DPMachine))

    def test_no_parent(self):
        with self.assertRaises(TypeError):
            IntToString()

    def test_empty_mechanism(self):
        mech = IntToString(ExponentialHierarchical())
        with self.assertRaises(ValueError):
            mech.randomise(1)

    def test_set_epsilon_locally(self):
        mech = IntToString(ExponentialHierarchical().set_hierarchy(["0", "1", "2", "3"]))
        mech.set_epsilon(1)
        self.assertIsNotNone(mech)

    def test_randomise(self):
        mech = IntToString(ExponentialHierarchical().set_hierarchy(["0", "1", "2", "3"]).set_epsilon(1))
        self.assertIsInstance(mech.randomise(1), int)

    def test_distrib(self):
        epsilon = np.log(2)
        runs = 10000
        mech = IntToString(ExponentialHierarchical().set_hierarchy(["0", "1", "2", "3"]).set_epsilon(epsilon))
        count = [0, 0]

        for _ in range(runs):
            val = mech.randomise(0)
            if val == 0:
                count[0] += 1

            val = mech.randomise(1)
            if val == 0:
                count[1] += 1

        self.assertGreater(count[0], count[1])
        # print("Sensitivity: %f" % mech.parent._sensitivity)
        # print("Balanced: %s" % str(mech.parent._balanced_tree))
        # print("Utilities: %s" % str(mech.parent.get_utility_list()))
        # print("Counts: 0: %d, 1: %d" % (count[0], count[1]))
        # print("Empirical epsilon: %f" % (count[0] / count[1]))
        # print("%f <= %f" % (count[0] / runs, count[1] * np.exp(epsilon) / runs))
        self.assertLessEqual(count[0] / runs, count[1] * np.exp(epsilon) / runs + 0.05)
