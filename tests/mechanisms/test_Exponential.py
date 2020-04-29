import numpy as np
from unittest import TestCase

from diffprivlib.mechanisms import Exponential
from diffprivlib.utils import global_seed


class TestExponential(TestCase):
    def setup_method(self, method):
        if method.__name__ .endswith("prob"):
            global_seed(314159)

        self.mech = Exponential()

    def teardown_method(self, method):
        del self.mech

    def test_not_none(self):
        self.assertIsNotNone(self.mech)

    def test_class(self):
        from diffprivlib.mechanisms import DPMechanism
        self.assertTrue(issubclass(Exponential, DPMechanism))

    def test_no_params(self):
        with self.assertRaises(ValueError):
            self.mech.randomise("A")

    def test_no_epsilon(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        self.mech.set_utility(utility_list)
        with self.assertRaises(ValueError):
            self.mech.randomise("A")

    def test_inf_epsilon(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        self.mech.set_utility(utility_list).set_epsilon(float("inf"))

        # print(_mech.randomise("A"))

        for i in range(1000):
            self.assertEqual(self.mech.randomise("A"), "A")

    def test_neg_epsilon(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon(-1)

    def test_complex_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon(1+2j)

    def test_string_epsilon(self):
        with self.assertRaises(TypeError):
            self.mech.set_epsilon("Two")

    def test_non_zero_delta(self):
        with self.assertRaises(ValueError):
            self.mech.set_epsilon_delta(1, 0.5)

    def test_no_utility(self):
        self.mech.set_epsilon(1)

        with self.assertRaises(ValueError):
            self.mech.randomise("1")

    def test_hierarchy_first(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "2", 2],
            ["B", "2", 2]
        ]
        self.mech.set_utility(utility_list)
        self.assertIsNotNone(self.mech)

    def test_non_string_hierarchy(self):
        utility_list = [
            ["A", "B", 1],
            ["A", 2, 2],
            ["B", 2, 2]
        ]
        with self.assertRaises(TypeError):
            self.mech.set_utility(utility_list)

    def test_missing_utilities(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2]
        ]
        with self.assertRaises(ValueError):
            self.mech.set_utility(utility_list)

    def test_wrong_utilities(self):
        utility_list = (
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        )
        with self.assertRaises(TypeError):
            self.mech.set_utility(utility_list)

        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", "2"]
        ]
        with self.assertRaises(TypeError):
            self.mech.set_utility(utility_list)

        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", -2]
        ]

        with self.assertRaises(ValueError):
            self.mech.set_utility(utility_list)

    def test_non_string_input(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        self.mech.set_epsilon(1).set_utility(utility_list)
        with self.assertRaises(TypeError):
            self.mech.randomise(2)

    def test_outside_domain(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        self.mech.set_epsilon(1).set_utility(utility_list)
        with self.assertRaises(ValueError):
            self.mech.randomise("D")

    def test_get_utility_list(self):
        self.assertIsNone(self.mech.get_utility_list())

        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        self.mech.set_epsilon(1).set_utility(utility_list)

        _utility_list = self.mech.get_utility_list()
        self.assertEqual(len(_utility_list), len(utility_list))

    def test_self_in_utility(self):
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2],
            ["A", "A", 5]
        ]
        self.mech.set_epsilon(1).set_utility(utility_list)

        _utility_list = self.mech.get_utility_list()
        self.assertEqual(len(_utility_list) + 1, len(utility_list))

        self.assertEqual(self.mech._get_utility("A", "A"), 0)

    def test_distrib_prob(self):
        epsilon = np.log(2)
        runs = 20000
        utility_list = [
            ["A", "B", 1],
            ["A", "C", 2],
            ["B", "C", 2]
        ]
        self.mech.set_epsilon(epsilon).set_utility(utility_list)
        count = [0, 0, 0]

        for i in range(runs):
            val = self.mech.randomise("A")
            if val == "A":
                count[0] += 1
            elif val == "B":
                count[1] += 1
            elif val == "C":
                count[2] += 1

        # print("A: %d, B: %d, C: %d" % (count[0], count[1], count[2]))
        self.assertLessEqual(count[0] / runs, np.exp(epsilon) * count[2] / runs + 0.05)
        self.assertAlmostEqual(count[0] / count[1], count[1] / count[2], delta=0.1)

    def test_repr(self):
        repr_ = repr(self.mech.set_epsilon(1))
        self.assertIn(".Exponential(", repr_)
