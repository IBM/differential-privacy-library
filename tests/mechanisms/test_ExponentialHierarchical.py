import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import ExponentialHierarchical
from diffprivlib.utils import global_seed


class TestExponentialHierarchical(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = ExponentialHierarchical()

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(ExponentialHierarchical, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise("A")

    def test_no_epsilon(self):
        self.mech.set_hierarchy([["A", "B"], ["C"]])
        with self.assertRaises(ValueError):
            self.mech.randomise("A")

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon(-1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon("Two")

    def test_inf_epsilon(self):
        self.mech.set_hierarchy([["A", "B"], ["C"]]).set_epsilon(float("inf"))

        for i in range(1000):
            self.assertEqual(self.mech.randomise("A"), "A")

    def test_non_zero_delta(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(1, 0.5)

    def test_hierarchy_first(self):
        self.mech.set_hierarchy([["A", "B"], ["C"]])
        self.assertIsNotNone(self.mech)

    def test_non_string_hierarchy(self):
        self.mech.set_epsilon(1)
        with self.assertRaises(TypeError):
            self.mech.set_hierarchy([["A", "B"], ["C", 2]])

    def test_non_list_hierarchy(self):
        with self.assertRaises(TypeError):
            self.mech.set_hierarchy(("A", "B", "C"))

    def test_uneven_hierarchy(self):
        with self.assertRaises(ValueError):
            self.mech.set_hierarchy(["A", ["B", "C"]])

    def test_build_utility_list(self):
        with self.assertRaises(TypeError):
            self.mech._build_utility_list([1, 2, 3])

    def test_non_string_input(self):
        self.mech.set_epsilon(1).set_hierarchy([["A", "B"], ["C", "2"]])
        with self.assertRaises(TypeError):
            self.mech.randomise(2)

    def test_outside_domain(self):
        self.mech.set_epsilon(1).set_hierarchy([["A", "B"], ["C"]])
        with self.assertRaises(ValueError):
            self.mech.randomise("D")

    def test_distrib_prob(self):
        epsilon = np.log(2)
        runs = 20000
        balanced_tree = False
        self.mech.set_epsilon(epsilon).set_hierarchy([["A", "B"], ["C"]])
        count = [0, 0, 0]

        for i in range(runs):
            val = self.mech.randomise("A")
            if val == "A":
                count[0] += 1
            elif val == "B":
                count[1] += 1
            elif val == "C":
                count[2] += 1

        # print(_mech.get_utility_list())
        # print(_mech._sensitivity)

        # print("A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertAlmostEqual(count[0] / runs, np.exp(epsilon / (1 if balanced_tree else 2)) * count[2] / runs, delta=0.1)
        self.assertAlmostEqual(count[0] / count[1], count[1] / count[2], delta=0.1)

    def test_neighbours_prob(self):
        epsilon = np.log(2)
        runs = 20000
        self.mech.set_epsilon(epsilon).set_hierarchy([["A", "B"], ["C"]])
        count = [0, 0, 0]

        for i in range(runs):
            val = self.mech.randomise("A")
            if val == "A":
                count[0] += 1

            val = self.mech.randomise("B")
            if val == "A":
                count[1] += 1

            val = self.mech.randomise("C")
            if val == "A":
                count[2] += 1

        # print("Output: A\nInput: A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs)
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[2] / runs)
        self.assertLessEqual(count[1] / runs, np.exp(epsilon) * count[2] / runs)

    def test_neighbours_flat_hierarchy_prob(self):
        epsilon = np.log(2)
        runs = 20000
        self.mech.set_epsilon(epsilon).set_hierarchy(["A", "B", "C"])
        count = [0, 0, 0]

        for i in range(runs):
            val = self.mech.randomise("A")
            if val == "A":
                count[0] += 1

            val = self.mech.randomise("B")
            if val == "A":
                count[1] += 1

            val = self.mech.randomise("C")
            if val == "A":
                count[2] += 1

        # print("(Output: A) Input: A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[1] / runs + 0.05)
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[2] / runs + 0.05)
        self.assertLessEqual(count[1] / runs, np.exp(epsilon) * count[2] / runs + 0.05)

    def test_repr(self):
        repr_ = repr(self.mech.set_epsilon(1))
        self.assertIn(".ExponentialHierarchical(", repr_)
