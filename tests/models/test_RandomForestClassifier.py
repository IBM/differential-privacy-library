import numpy as np
from unittest import TestCase
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from diffprivlib.models.forest import RandomForestClassifier
from diffprivlib.utils import PrivacyLeakWarning, global_seed, BudgetError, DiffprivlibCompatibilityWarning


class TestRandomForestClassifier(TestCase):
    def setUp(self):
        global_seed(2718281828)
    
    def test_not_none(self):
        self.assertIsNotNone(RandomForestClassifier)

    def test_bad_params(self):
        X = [[1]]
        y = [0]

        with self.assertRaises(TypeError):
            RandomForestClassifier(n_estimators="10").fit(X, y)
        
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            RandomForestClassifier(n_estimators=1, bounds=([0], [1])).fit(X, y, sample_weight=1)

    def test_bad_data(self):
        with self.assertRaises(ValueError):
            RandomForestClassifier(bounds=([0], [1])).fit([[1]], None)

        with self.assertRaises(ValueError):
            RandomForestClassifier(bounds=([0], [1])).fit([[1], [2]], [1])

        with self.assertRaises(ValueError):
            RandomForestClassifier(bounds=([0], [1])).fit([[1], [2]], [[1, 2], [2, 4]])

        with self.assertRaises(ValueError):
            RandomForestClassifier(bounds=([0], [1])).fit([[1, 2], [3, 4]], [1, 0])

    def test_multi_output(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)

        clf = RandomForestClassifier()
        with self.assertRaises(ValueError):
            clf.fit(X, [y, y])

    def test_simple(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = RandomForestClassifier(epsilon=5, n_estimators=5, random_state=2021)

        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        # when `feature_domains` is not provided, we should get a privacy leakage warning

        with self.assertWarns(PrivacyLeakWarning):
            model.fit(X, y)
        check_is_fitted(model)

        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_classes_, 2)
        self.assertEqual(set(model.classes_), set([0, 1]))
        self.assertEqual(model.max_depth_, 3)
        self.assertTrue(model.estimators_)
        self.assertEqual(len(model.estimators_), 5)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))
        self.assertTrue(np.all(model.bounds[0] == [2, 3, 4]))
        self.assertTrue(np.all(model.bounds[1] == [12, 13, 15]))

    def test_with_bounds(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = RandomForestClassifier(epsilon=5, n_estimators=5, random_state=2021, bounds=([2, 3, 4], [12, 13, 15]))
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        model.fit(X, y)
        check_is_fitted(model)
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_classes_, 2)
        self.assertEqual(set(model.classes_), set([0, 1]))
        self.assertEqual(model.max_depth_, 3)
        self.assertTrue(model.estimators_)
        self.assertEqual(len(model.estimators_), 5)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))
        self.assertTrue(np.all(model.bounds[0] == [2, 3, 4]))
        self.assertTrue(np.all(model.bounds[1] == [12, 13, 15]))

    def test_with_not_enough_bounds(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, random_state=2021, bounds=([2, 3], [12, 13]))
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant()

        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        bounds = ([2, 3, 4], [12, 13, 15])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, accountant=acc, bounds=bounds)
        model.fit(X, y)
        self.assertEqual((2, 0), acc.total())
        
        with BudgetAccountant(3, 0) as acc2:
            model = RandomForestClassifier(epsilon=2, n_estimators=5, bounds=bounds)
            model.fit(X, y)
            self.assertEqual((2, 0), acc2.total())

            with self.assertRaises(BudgetError):
                model.fit(X, y)
