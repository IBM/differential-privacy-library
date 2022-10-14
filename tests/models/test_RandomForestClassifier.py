import numpy as np
from unittest import TestCase

from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from diffprivlib.models.forest import RandomForestClassifier, DecisionTreeClassifier
from diffprivlib.utils import PrivacyLeakWarning, BudgetError, DiffprivlibCompatibilityWarning

X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]] * 3
y = [-1, -1, -1, 1, 1, 1] * 3
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]


class TestRandomForestClassifier(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(RandomForestClassifier)

    def test_bad_params(self):
        X = [[1]]
        y = [0]

        with self.assertRaises((ValueError, TypeError)):  # Should be TypeError?
            RandomForestClassifier(n_estimators="10", bounds=(0, 1), classes=[0, 1]).fit(X, y)

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            RandomForestClassifier(n_estimators=1, bounds=(0, 1), classes=[0, 1]).fit(X, y, sample_weight=1)

    def test_bad_data(self):
        with self.assertRaises(ValueError):
            RandomForestClassifier(bounds=([0], [1])).fit([[1]], None)

        with self.assertRaises(ValueError):
            RandomForestClassifier(bounds=([0], [1])).fit([[1], [2]], [1])

        with self.assertRaises(ValueError):
            RandomForestClassifier(bounds=([0], [1])).fit([[1], [2]], [[1, 2], [2, 4]])

        # with self.assertRaises(ValueError):
        #     RandomForestClassifier(bounds=([0], [1])).fit([[1, 2], [3, 4]], [1, 0])

    def test_multi_output(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)

        clf = RandomForestClassifier()
        with self.assertRaises(ValueError):
            clf.fit(X, [y, y])

    def test_simple(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = RandomForestClassifier(epsilon=5, n_estimators=5, max_depth=3, random_state=1)

        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

        # when `bounds` is not provided, we should get a privacy leakage warning
        with self.assertWarns(PrivacyLeakWarning):
            model.fit(X, y)
        check_is_fitted(model)

        # Test parameters and attributes
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_classes_, 2)
        self.assertEqual(set(model.classes_), set([0, 1]))
        self.assertEqual(model.max_depth, 3)
        self.assertTrue(model.estimators_)
        self.assertEqual(len(model.estimators_), 5)

        self.assertTrue(model.predict(np.array([[12, 3, 14]])))
        self.assertTrue(np.all(model.bounds[0] == [2, 3, 4]))
        self.assertTrue(np.all(model.bounds[1] == [12, 13, 15]))

    def test_with_bounds(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3)
        model = RandomForestClassifier(epsilon=5, n_estimators=5, bounds=([2, 3, 4], [12, 13, 15]), classes=[0, 1],
                                       max_depth=3, random_state=1)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

        model.fit(X, y)

        check_is_fitted(model)

        # Test parameters and attributes
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_classes_, 2)
        self.assertEqual(set(model.classes_), set([0, 1]))
        self.assertEqual(model.max_depth, 3)
        self.assertTrue(model.estimators_)
        self.assertEqual(len(model.estimators_), 5)
        self.assertTrue(model.predict(np.array([[12, 3, 14]])))
        self.assertTrue(np.all(model.bounds[0] == [2, 3, 4]))
        self.assertTrue(np.all(model.bounds[1] == [12, 13, 15]))

    def test_non_binary_classes(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]] * 3)
        y = np.array([1, 1, 1, 0, 0, 0, 1] * 3) + 2
        model = RandomForestClassifier(epsilon=5, n_estimators=5, bounds=([2, 3, 4], [12, 13, 15]), classes=[2, 3],
                                       max_depth=3)
        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

        model.fit(X, y)

        self.assertIn(model.predict(np.array([[12, 3, 14]])), (2, 3))
        self.assertIn(model.predict(np.array([[2, 13, 4]])), (2, 3))
        self.assertIn(model.predict(np.array([[3, 5, 15]])), (2, 3))

    def test_with_not_enough_bounds(self):
        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, bounds=([2, 3], [12, 13]), classes=[0, 1])

        with self.assertRaises(NotFittedError):
            check_is_fitted(model)

        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_random_state(self):
        model0 = RandomForestClassifier(20, epsilon=0.1, bounds=(-2, 2), classes=[-1, 1], max_depth=3,
                                        random_state=0)
        model1 = RandomForestClassifier(20, epsilon=0.1, bounds=(-2, 2), classes=[-1, 1], max_depth=3,
                                        random_state=2)
        model0.fit(X, y)
        model1.fit(X, y)

        self.assertFalse(np.isclose(model0.predict_proba(np.array([[-2, -1]])),
                                    model1.predict_proba(np.array([[-2, -1]]))).any(),
                         (model0.predict_proba(np.array([[-2, -1]])), model1.predict_proba(np.array([[-2, -1]]))))

        model1 = RandomForestClassifier(20, epsilon=0.1, bounds=(-2, 2), classes=[-1, 1], max_depth=3,
                                        random_state=0)
        model1.fit(X, y)
        for i in range(20):
            self.assertIsNotNone(model0.estimators_[i].random_state)
            self.assertEqual(model0.estimators_[i].random_state, model1.estimators_[i].random_state)

        self.assertTrue(np.isclose(model0.predict_proba(np.array([[-2, -1]])),
                                   model1.predict_proba(np.array([[-2, -1]]))).any(),
                        (model0.predict_proba(np.array([[-2, -1]])), model1.predict_proba(np.array([[-2, -1]]))))

    def test_sklearn_methods(self):
        depth = 3
        n_estimators = 4
        clf = RandomForestClassifier(n_estimators, epsilon=5, bounds=(-2, 2), classes=[-1, 1],
                                     max_depth=depth)
        clf.fit(X, y)

        self.assertEqual(clf.n_features_in_, 2)

        self.assertIsInstance(clf.apply(T), np.ndarray)
        self.assertEqual(clf.apply(T).shape, (len(T), n_estimators))

        for estimator in clf:
            self.assertIsInstance(estimator, DecisionTreeClassifier)

        self.assertEqual(len(clf), n_estimators)
        self.assertIsInstance([e for e in clf], list)

        self.assertIsInstance(clf.score(T, true_result), float)
        self.assertTrue(0 <= clf.score(T, true_result) <= 1)

        self.assertIsInstance(clf.decision_path(X), tuple)
        self.assertTrue(issparse(clf.decision_path(X)[0]), msg=str(clf.decision_path(X)))

        self.assertIsInstance(clf.predict(T), np.ndarray)
        self.assertEqual(clf.predict(T).shape, (len(T),))

        self.assertIsInstance(clf.predict_proba(T), np.ndarray)
        self.assertEqual(clf.predict_proba(T).shape, (len(T), clf.n_classes_))

        self.assertEqual(clf.ccp_alpha, 0.0)
        self.assertIsNone(clf.class_weight)

    def test_warm_start(self):
        depth = 3
        n_estimators = 4
        clf = RandomForestClassifier(n_estimators, epsilon=1, bounds=(-2, 2), classes=[-1, 1],
                                     max_depth=depth, warm_start=True, random_state=0)
        clf.fit(X, y)
        first_estimator = clf[0].__dict__.copy()

        clf.n_estimators = n_estimators * 2
        clf.fit(X, y)

        self.assertEqual(len(clf), n_estimators * 2)
        self.assertEqual(clf[0].__dict__, first_estimator)
        self.assertNotEqual(clf[n_estimators].__dict__, first_estimator)

        with self.assertWarns(UserWarning):
            clf.fit(X, y)

        clf.n_estimators = n_estimators
        with self.assertRaises(ValueError):
            clf.fit(X, y)

        # Estimators should still have the same seed
        clf2 = RandomForestClassifier(n_estimators * 2, epsilon=1, bounds=(-2, 2), classes=[-1, 1],
                                      max_depth=depth, warm_start=True, random_state=0)
        clf2.fit(X, y)

        for i in range(n_estimators * 2):
            self.assertEqual(clf.estimators_[i].random_state, clf2.estimators_[i].random_state)

    def test_parallel(self):
        depth = 3
        n_estimators = 4
        clf = RandomForestClassifier(n_estimators, epsilon=5, bounds=(-2, 2), classes=[-1, 1],
                                     max_depth=depth, n_jobs=-1)
        clf.fit(X, y)

        self.assertIsNone(check_is_fitted(clf))

    def test_shuffle(self):
        depth = 3
        n_estimators = 4
        clf = RandomForestClassifier(n_estimators, epsilon=5, bounds=(-2, 2), classes=[-1, 1],
                                     max_depth=depth, random_state=0, shuffle=False)
        clf.fit(X, y)

        clf_shuffle = RandomForestClassifier(n_estimators, epsilon=5, bounds=(-2, 2), classes=[-1, 1],
                                             max_depth=depth, random_state=0, shuffle=True)
        clf_shuffle.fit(X, y)
        self.assertIsNone(check_is_fitted(clf_shuffle))

        # Estimators should have the same seed
        for i in range(n_estimators):
            self.assertEqual(clf.estimators_[i].random_state, clf_shuffle.estimators_[i].random_state)

        # But predictions should be different, since trained on different data
        self.assertFalse(np.isclose(clf.predict_proba(T), clf_shuffle.predict_proba(T)).all(),
                         (clf.predict_proba(T), clf_shuffle.predict_proba(T)))

    def test_more_estimators_than_data(self):
        depth = 3
        n_estimators = 20
        clf = RandomForestClassifier(n_estimators, epsilon=5, bounds=(-2, 2), classes=[-1, 1],
                                     max_depth=depth)
        clf.fit(X, y)

        self.assertIsNone(check_is_fitted(clf))

        self.assertIsInstance(clf.predict(T), np.ndarray)
        self.assertEqual(clf.predict(T).shape, (len(T),))
        self.assertEqual(clf.predict_proba(T).shape, (len(T), clf.n_classes_))

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant()

        X = np.array([[12, 3, 14], [12, 3, 4], [12, 3, 4], [2, 13, 4], [2, 13, 14], [2, 3, 14], [3, 5, 15]])
        y = np.array([1, 1, 1, 0, 0, 0, 1])
        bounds = ([2, 3, 4], [12, 13, 15])
        model = RandomForestClassifier(epsilon=2, n_estimators=5, accountant=acc, bounds=bounds, classes=[0, 1])
        model.fit(X, y)
        self.assertEqual((2, 0), acc.total())

        with BudgetAccountant(3, 0) as acc2:
            model = RandomForestClassifier(epsilon=2, n_estimators=5, bounds=bounds, classes=[0, 1])
            model.fit(X, y)
            self.assertEqual((2, 0), acc2.total())

            with self.assertRaises(BudgetError):
                model.fit(X, y)
