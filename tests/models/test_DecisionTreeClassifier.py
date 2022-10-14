import numpy as np
from unittest import TestCase

from scipy.sparse import issparse
from sklearn import datasets
from sklearn.tree._tree import Tree
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from diffprivlib.models.forest import DecisionTreeClassifier
from diffprivlib.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning, BudgetError


X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]
iris = datasets.load_iris()


class TestDecisionTreeClassifier(TestCase):
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

    def test_simple_prob(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = DecisionTreeClassifier(epsilon=5, max_depth=3, random_state=0)
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
        model = DecisionTreeClassifier(epsilon=5, bounds=([2, 3, 4], [12, 13, 14]), classes=[0, 1], max_depth=3,
                                       random_state=0)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        model.fit(X, y)
        check_is_fitted(model)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))

    def test_with_non_binary_labels_prob(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([3, 3, 3, 3, 5, 5, 3] * 3)
        model = DecisionTreeClassifier(epsilon=5, bounds=([2, 3, 4], [12, 13, 14]), classes=[3, 5], max_depth=3,
                                       random_state=0)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)
        model.fit(X, y)
        check_is_fitted(model)
        self.assertEqual(model.predict(np.array([[12, 3, 14]])), 3)

    def test_random_state(self):
        model0 = DecisionTreeClassifier(epsilon=0.01, bounds=(-2, 2), classes=[-1, 1], max_depth=3, random_state=0)
        model1 = DecisionTreeClassifier(epsilon=0.01, bounds=(-2, 2), classes=[-1, 1], max_depth=3, random_state=2)
        model0.fit(X, y)
        model1.fit(X, y)
        self.assertNotEqual(model0.predict(np.array([[-2, -1]])), model1.predict(np.array([[-2, -1]])))

        model1 = DecisionTreeClassifier(epsilon=0.01, bounds=(-2, 2), classes=[-1, 1], max_depth=3, random_state=0)
        model1.fit(X, y)
        self.assertTrue(np.isclose(model0.predict_proba(np.array([[-2, -1]])),
                                   model1.predict_proba(np.array([[-2, -1]]))).all())

        model1 = DecisionTreeClassifier(epsilon=0.01, bounds=(-2, 2), classes=[-1, 1], max_depth=3, random_state=None)
        model1.fit(X, y)

    def test_sklearn_methods(self):
        depth = 3
        clf = DecisionTreeClassifier(epsilon=5, bounds=(-2, 2), classes=[-1, 1], max_depth=depth)
        clf.fit(X, y)

        self.assertEqual(clf.n_features_in_, 2)

        self.assertEqual(clf.get_depth(), depth)

        self.assertEqual(clf.get_n_leaves(), 2 ** depth)

        self.assertIsInstance(clf.apply(T), np.ndarray)
        self.assertEqual(clf.apply(T).shape, (len(T),))

        self.assertIsInstance(clf.tree_, Tree)

        self.assertIsInstance(clf.score(T, true_result), float)
        self.assertTrue(0 <= clf.score(T, true_result) <= 1)

        self.assertTrue(issparse(clf.decision_path(X)))

        self.assertIsNone(clf._prune_tree())

        self.assertIsInstance(clf.predict(T), np.ndarray)
        self.assertEqual(clf.predict(T).shape, (len(T),))

        self.assertIsInstance(clf.predict_proba(T), np.ndarray)
        self.assertEqual(clf.predict_proba(T).shape, (len(T), clf.n_classes_))

        self.assertEqual(clf.ccp_alpha, 0.0)
        self.assertIsNone(clf.class_weight)

        self.assertIsNotNone(clf.n_features_)

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant()

        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        bounds = ([2, 3, 4], [12, 13, 15])
        model = DecisionTreeClassifier(epsilon=2, bounds=bounds, classes=[0, 1], max_depth=3, accountant=acc)
        model.fit(X, y)
        self.assertEqual((2, 0), acc.total())

        with BudgetAccountant(3, 0) as acc2:
            model = DecisionTreeClassifier(epsilon=2, bounds=bounds, classes=[0, 1], max_depth=3)
            model.fit(X, y)
            self.assertEqual((2, 0), acc2.total())

            with self.assertRaises(BudgetError):
                model.fit(X, y)
