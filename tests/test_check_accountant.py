from unittest import TestCase

from diffprivlib.accountant import BudgetAccountant, check_accountant


class TestCheck_accountant(TestCase):
    def test_simple(self):
        acc = BudgetAccountant()
        self.assertIsNone(check_accountant(acc))

    def test_fail(self):
        acc = {"epsilon": 1, "delta": 0.5}
        with self.assertRaises(TypeError):
            check_accountant(acc)
