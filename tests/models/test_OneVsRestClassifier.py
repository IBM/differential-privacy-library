# pylint: disable=C0114,C0115,C0116,R0904

import warnings
from unittest import TestCase, skipIf

import numpy as np
from sklearn import datasets
import sklearn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import parse_version

from diffprivlib.models import LogisticRegression, OneVsRestClassifier, GaussianNB, RandomForestClassifier
from diffprivlib.utils import PrivacyLeakWarning

iris = datasets.load_iris()
rng = np.random.RandomState(0)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]
_N_CLASSES = 3


class _ZeroDecision(BaseEstimator):
    def fit(self, X, y):
        return self

    def decision_function(self, X):
        return np.zeros(len(X))


class TestOneVsRestClassifier(TestCase):

    def test_not_none(self):
        self.assertIsNotNone(OneVsRestClassifier)

    def test_no_params(self):
        X = [[1]]
        y = [0]
        clf = OneVsRestClassifier(DummyClassifier(), classes=[0])
        clf.fit(X, y)

    def test_not_fitted_raises(self):
        """Predicting before fit must raise NotFittedError."""
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        with self.assertRaises(NotFittedError):
            ovr.predict([])

    def test_multioutput_target_raises(self):
        """2-D non-indicator targets must raise ValueError."""
        X = np.array([[1, 0], [0, 1]])
        y = np.array([[1, 2], [3, 1]])
        with self.assertRaises(ValueError):
            OneVsRestClassifier(DummyClassifier(), classes=[1, 2, 3]).fit(X, y)

    def test_privacy_leak_warning_without_classes(self):
        """Omitting `classes` must emit PrivacyLeakWarning."""
        ovr = OneVsRestClassifier(DummyClassifier())
        X = iris.data
        y = iris.target
        with self.assertWarns(PrivacyLeakWarning):
            ovr.fit(X, y)

    def test_no_privacy_leak_warning_with_classes(self):
        """Supplying `classes` must not emit PrivacyLeakWarning."""
        ovr = OneVsRestClassifier(DummyClassifier(), classes=[0, 1, 2])
        X = iris.data
        y = iris.target
        with warnings.catch_warnings():
            warnings.simplefilter("error", PrivacyLeakWarning)
            ovr.fit(X, y)  # must not raise

    def test_n_estimators_equals_n_classes_sklearn_base(self):
        """One sub-estimator per class must be fitted (non-DP base)."""
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        ovr.fit(iris.data, iris.target)
        self.assertEqual(len(ovr.estimators_), _N_CLASSES)

    def test_n_estimators_equals_n_classes_dp_base(self):
        """One sub-estimator per class must be fitted (DP base)."""
        ovr = OneVsRestClassifier(
            LogisticRegression(data_norm=10.0, epsilon=2.0), classes=[0, 1, 2]
        )
        ovr.fit(iris.data, iris.target)
        self.assertEqual(len(ovr.estimators_), _N_CLASSES)

    def test_fit_predict_sklearn_base(self):
        """OvR with a non-DP base estimator must produce valid predictions."""
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        pred = ovr.fit(iris.data, iris.target).predict(iris.data)
        self.assertEqual(pred.shape, iris.target.shape)
        self.assertGreater(np.mean(iris.target == pred), 0.65)

    def test_fit_predict_dp_base(self):
        """OvR with DP LogisticRegression must produce valid predictions."""
        ovr = OneVsRestClassifier(
            LogisticRegression(data_norm=10.0, epsilon=5.0, random_state=0),
            classes=[0, 1, 2],
        )
        pred = ovr.fit(iris.data, iris.target).predict(iris.data)
        self.assertEqual(pred.shape, iris.target.shape)
        # DP noise means lower accuracy bar; still expect better than random (1/3)
        self.assertGreater(np.mean(iris.target == pred), 0.35)

    def test_fit_predict_proba_sklearn_base(self):
        """predict_proba must sum to 1 per sample for single-label problems."""
        ovr = OneVsRestClassifier(
            skLogisticRegression(solver="lbfgs", max_iter=200, random_state=0),
            classes=[0, 1, 2],
        )
        ovr.fit(iris.data, iris.target)
        proba = ovr.predict_proba(iris.data)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_regressor_as_base_estimator(self):
        """A regressor base works and predicts all three classes."""
        ovr = OneVsRestClassifier(DecisionTreeRegressor(), classes=[0, 1, 2])
        pred = ovr.fit(iris.data, iris.target).predict(iris.data)
        self.assertEqual(len(ovr.estimators_), _N_CLASSES)
        assert_array_equal(np.unique(pred), [0, 1, 2])
        self.assertGreater(np.mean(pred == iris.target), 0.9)

    def test_classes_attribute_set_from_constructor(self):
        ovr = OneVsRestClassifier(DummyClassifier(), classes=[0, 1, 2])
        ovr.fit(iris.data, iris.target)
        assert_array_equal(ovr.classes_, [0, 1, 2])

    def test_classes_attribute_inferred_from_data_with_warning(self):
        ovr = OneVsRestClassifier(DummyClassifier())
        with self.assertWarns(PrivacyLeakWarning):
            ovr.fit(iris.data, iris.target)
        assert_array_equal(np.sort(ovr.classes_), [0, 1, 2])

    def test_n_classes_attribute(self):
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        ovr.fit(iris.data, iris.target)
        self.assertEqual(ovr.n_classes_, _N_CLASSES)

    def test_n_features_in_propagated(self):
        """n_features_in_ of the OvR must equal the number of input features."""
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        ovr.fit(iris.data, iris.target)
        self.assertEqual(ovr.n_features_in_, iris.data.shape[1])

    def test_string_class_labels(self):
        X = np.array([[0, 0, 5], [0, 5, 0], [3, 0, 0], [0, 0, 6], [6, 0, 0]])
        y = ["eggs", "spam", "ham", "eggs", "ham"]
        classes = ["eggs", "ham", "spam"]
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=classes)
        ovr.fit(X, y)
        y_pred = ovr.predict(np.array([[0, 0, 4]]))[0]
        self.assertEqual(y_pred, "eggs")

    def test_binary_classification(self):
        X = np.array([[0, 0, 5], [0, 5, 0], [3, 0, 0], [0, 0, 6], [6, 0, 0]])
        y = ["eggs", "spam", "spam", "eggs", "spam"]
        ovr = OneVsRestClassifier(
            LinearSVC(random_state=0), classes=["eggs", "spam"]
        )
        ovr.fit(X, y)
        y_pred = ovr.predict(np.array([[0, 0, 4]]))[0]
        self.assertEqual(y_pred, "eggs")

    def test_pipeline_base_estimator(self):
        """OvR must work when the base estimator is a Pipeline."""
        clf = Pipeline([("tree", DecisionTreeClassifier())])
        ovr_pipe = OneVsRestClassifier(clf, classes=[0, 1, 2])
        ovr_pipe.fit(iris.data, iris.target)
        ovr = OneVsRestClassifier(DecisionTreeClassifier(), classes=[0, 1, 2])
        ovr.fit(iris.data, iris.target)
        assert_array_equal(
            ovr.predict(iris.data), ovr_pipe.predict(iris.data)
        )

    def test_partial_fit_not_available_for_svc(self):
        """partial_fit must not be surfaced when base has no partial_fit."""
        ovr = OneVsRestClassifier(SVC(), classes=[0, 1, 2])
        self.assertFalse(hasattr(ovr, "partial_fit"))

    def test_gridsearch_compatible(self):
        """OvR should be usable inside GridSearchCV."""
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        Cs = [0.1, 0.5, 0.8]
        cv = GridSearchCV(ovr, {"estimator__C": Cs})
        cv.fit(iris.data, iris.target)
        best_C = cv.best_estimator_.estimators_[0].C
        self.assertIn(best_C, Cs)

    def test_epsilon_property_exposed(self):
        """epsilon property must reflect the base estimator's epsilon."""
        base = LogisticRegression(data_norm=10.0, epsilon=3.0)
        ovr = OneVsRestClassifier(base, classes=[0, 1, 2])
        self.assertEqual(ovr.epsilon, 3.0)

    def test_epsilon_property_none_for_non_dp(self):
        """epsilon property must return None for a non-DP base estimator."""
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        self.assertIsNone(ovr.epsilon)

    def test_delta_property_none_for_non_dp(self):
        """delta property must return None for a non-DP base estimator."""
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        self.assertIsNone(ovr.delta)

    def test_epsilon_constructor_sets_base_epsilon(self):
        """Passing epsilon to OvR constructor must propagate to base DP model."""
        base = LogisticRegression(data_norm=10.0)
        ovr = OneVsRestClassifier(base, classes=[0, 1, 2], epsilon=7.0)
        self.assertEqual(ovr.estimator.epsilon, 7.0)

    def test_decision_function_shape_multiclass(self):
        """decision_function must return shape (n_samples, n_classes)."""
        ovr = OneVsRestClassifier(LinearSVC(random_state=0), classes=[0, 1, 2])
        ovr.fit(iris.data, iris.target)
        df = ovr.decision_function(iris.data)
        self.assertEqual(df.shape, (iris.data.shape[0], _N_CLASSES))

    @skipIf(parse_version(sklearn.__version__) < parse_version("1.8.0"),
            "scikit-learn < 1.8.0 breaks ties differently from np.argmax")
    def test_ties_broken_by_argmax(self):
        """Tie-breaking in predict must match np.argmax(decision_function)."""

        X = np.array([[0], [0], [0], [0]])
        y = np.array([0, 1, 2, 3])
        clf = OneVsRestClassifier(_ZeroDecision(), classes=[0, 1, 2, 3])
        clf.fit(X, y)
        assert_array_equal(
            clf.predict(X), np.argmax(clf.decision_function(X), axis=1)
        )

    def test_score_method(self):
        """score() must return a value in [0, 1]."""
        ovr = OneVsRestClassifier(SVC(), classes=[0, 1, 2])
        ovr.fit(iris.data, iris.target)
        score = ovr.score(iris.data, iris.target)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.9)

    def test_dp_gaussiannb_fit_predict(self):
        bounds = ([4.0, 2.0, 1.0, 0.0], [8.0, 5.0, 7.0, 3.0])
        base = GaussianNB(epsilon=2.0, bounds=bounds, random_state=0)
        ovr = OneVsRestClassifier(base, classes=[0, 1, 2])

        ovr.fit(iris.data, iris.target)

        # Correct number of sub-estimators
        self.assertEqual(len(ovr.estimators_), _N_CLASSES)

        # Predictions have the right shape and contain only known class labels
        pred = ovr.predict(iris.data)
        self.assertEqual(pred.shape, iris.target.shape)
        self.assertTrue(set(np.unique(pred)).issubset({0, 1, 2}))

        # predict_proba must be available (GaussianNB exposes it) and sum to 1
        proba = ovr.predict_proba(iris.data)
        self.assertEqual(proba.shape, (iris.data.shape[0], _N_CLASSES))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

        # With high epsilon the accuracy should comfortably beat random chance
        self.assertGreater(np.mean(pred == iris.target), 0.35)

    def test_dp_random_forest_fit_predict(self):
        # bounds cover the iris feature ranges; base sees binary {0,1} problems
        bounds = ([4.0, 2.0, 1.0, 0.0], [8.0, 5.0, 7.0, 3.0])
        base = RandomForestClassifier(
            n_estimators=5, epsilon=2.0, bounds=bounds, classes=[0, 1],
            max_depth=3, random_state=0,
        )
        ovr = OneVsRestClassifier(base, classes=[0, 1, 2])

        ovr.fit(iris.data, iris.target)

        self.assertEqual(len(ovr.estimators_), _N_CLASSES)

        pred = ovr.predict(iris.data)
        self.assertEqual(pred.shape, iris.target.shape)
        self.assertTrue(set(np.unique(pred)).issubset({0, 1, 2}))

        proba = ovr.predict_proba(iris.data)
        self.assertEqual(proba.shape, (iris.data.shape[0], _N_CLASSES))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

        self.assertGreater(np.mean(pred == iris.target), 0.35)
