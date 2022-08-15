import numpy as np
from unittest import TestCase
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from diffprivlib.models.forest import DecisionTreeClassifier, calc_tree_depth
from diffprivlib.utils import PrivacyLeakWarning, global_seed, DiffprivlibCompatibilityWarning


class TestDecisionTreeClassifier(TestCase):
    def setUp(self):
        global_seed(2718281828)

    def test_not_none(self):
        self.assertIsNotNone(DecisionTreeClassifier)

    def test_bad_params(self):
        X = [[1]]
        y = [0]
        
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(epsilon=-1, bounds=([0], [1])).fit(X, y)

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            DecisionTreeClassifier(bounds=([0], [1])).fit(X, y, sample_weight=1)

    def test_bad_data(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(bounds=([0], [1])).fit([[1]], None)

        with self.assertRaises(ValueError):
            DecisionTreeClassifier(bounds=([0], [1])).fit([[1], [2]], [1])

        with self.assertRaises(ValueError):
            DecisionTreeClassifier(bounds=([0], [1])).fit([[1], [2]], [[1, 2], [2, 4]])

    def test_simple(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = DecisionTreeClassifier(epsilon=5, max_depth=3)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        # when `bounds` is not provided, we should get a privacy leakage warning
        with self.assertWarns(PrivacyLeakWarning):
            model.fit(X, y)
        check_is_fitted(model)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))

    def test_with_bounds(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = DecisionTreeClassifier(epsilon=5, bounds=([2, 3, 4], [12, 13, 14]), max_depth=3)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        model.fit(X, y)
        check_is_fitted(model)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))

    def test_with_non_binary_labels(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([3, 3, 3, 3, 5, 5, 3] * 3)
        model = DecisionTreeClassifier(epsilon=5, bounds=([2, 3, 4], [12, 13, 14]), max_depth=3)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        model.fit(X, y)
        check_is_fitted(model)
        self.assertEqual(model.predict(np.array([[12, 3, 14]])), 3)
