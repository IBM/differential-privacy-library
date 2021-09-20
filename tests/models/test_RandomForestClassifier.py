import numpy as np
import pytest
from unittest import TestCase
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from diffprivlib.models.forest import RandomForestClassifier
from diffprivlib.utils import PrivacyLeakWarning, global_seed, BudgetError


@pytest.mark.filterwarnings('ignore::diffprivlib.utils.PrivacyLeakWarning',
                            'ignore::sklearn.utils.validation.DataConversionWarning')
class TestRandomForestClassifier(TestCase):
    def setUp(self):
        global_seed(2718281828)
    
    def test_not_none(self):
        self.assertIsNotNone(RandomForestClassifier)

    def test_bad_params(self):
        X = [[1]]
        y = [0]

        with self.assertRaises(ValueError):
            RandomForestClassifier(n_estimators="10").fit(X, y)
        
        with self.assertRaises(ValueError):
            RandomForestClassifier(cat_feature_threshold="5").fit(X, y)

    def test_bad_data(self):
        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([[1]], None)

        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([[1], [2]], [1])

        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([[1], [2]], [[1, 2], [2, 4]])

        with self.assertRaises(ValueError):
            RandomForestClassifier().fit([[1, 2], [3, 4]], [1, 0])

    def test_simple(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=5, n_estimators=5, random_state=2021, cat_feature_threshold=2)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        # when `feature_domains` is not provided, we should get a privacy leakage warning
        with self.assertWarns(PrivacyLeakWarning):
            model.fit(X, y)
        check_is_fitted(model)
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_classes_, 2)
        self.assertEqual(set(model.classes_), set([0, 1]))
        self.assertEqual(model.cat_features_, [])
        self.assertEqual(model.max_depth_, 3)
        self.assertTrue(model.estimators_)
        self.assertEqual(len(model.estimators_), 5)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))
        self.assertIsNone(model.feature_domains)
        self.assertEqual(model.feature_domains_, {'0': [2.0, 12.0], '1': [3.0, 13.0], '2': [4.0, 15.0]})

    def test_with_feature_domains(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=5, n_estimators=5, random_state=2021, cat_feature_threshold=2,
                                       feature_domains={'0': [2.0, 12.0], '1': [3.0, 13.0], '2': [4.0, 15.0]})
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        model.fit(X, y)
        check_is_fitted(model)
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_classes_, 2)
        self.assertEqual(set(model.classes_), set([0, 1]))
        self.assertEqual(model.cat_features_, [])
        self.assertEqual(model.max_depth_, 3)
        self.assertTrue(model.estimators_)
        self.assertEqual(len(model.estimators_), 5)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))
        self.assertEqual(model.feature_domains_, {'0': [2.0, 12.0], '1': [3.0, 13.0], '2': [4.0, 15.0]})

    def test_with_not_enough_feature_domains(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, random_state=2021, feature_domains={'0': [2.0, 12.0], '1': [3.0, 13.0]})
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant()

        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, accountant=acc)
        model.fit(X, y)
        self.assertEqual((2, 0), acc.total())
        
        with BudgetAccountant(3, 0) as acc2:
            model = RandomForestClassifier(epsilon=2, n_estimators=5)
            model.fit(X, y)
            self.assertEqual((2, 0), acc2.total())

            with self.assertRaises(BudgetError):
                model.fit(X, y)
