from unittest import TestCase

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from diffprivlib.models.naive_bayes import GaussianNB
from diffprivlib.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning, BudgetError, check_random_state

# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([1, 1, 1, 2, 2, 2])


class TestGaussianNB(TestCase):
    def test_not_none(self):
        clf = GaussianNB(epsilon=1, bounds=(-2, 2))
        self.assertIsNotNone(clf)

    def test_zero_epsilon(self):
        clf = GaussianNB(epsilon=0, bounds=(-2, 2))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_neg_epsilon(self):
        clf = GaussianNB(epsilon=-1, bounds=(-2, 2))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_sample_weight_warning(self):
        clf = GaussianNB(epsilon=1, bounds=(-2, 2))
        w = abs(np.random.randn(10))

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            clf.fit(X, y, sample_weight=w)

    def test_mis_ordered_bounds(self):
        clf = GaussianNB(epsilon=1, bounds=([0, 1], [1, 0]))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_no_bounds(self):
        clf = GaussianNB()

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

        self.assertIsNotNone(clf)

    def test_missing_bounds(self):
        rng = check_random_state(0)
        X = rng.random((10, 6))

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_noisy_count(self):
        rng = check_random_state(0)
        y = rng.randint(20, size=10000)
        actual_counts = np.array([(y == y_i).sum() for y_i in np.unique(y)])

        clf = GaussianNB(epsilon=3)
        noisy_counts = clf._noisy_class_counts(y, random_state=None)
        self.assertEqual(y.shape[0], noisy_counts.sum())
        self.assertFalse(np.all(noisy_counts == actual_counts))

        clf = GaussianNB(epsilon=float("inf"))
        noisy_counts = clf._noisy_class_counts(y, random_state=None)
        self.assertEqual(y.shape[0], noisy_counts.sum())
        self.assertTrue(np.all(noisy_counts == actual_counts))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_different_results(self):
        from sklearn.naive_bayes import GaussianNB as sk_nb
        from sklearn import datasets

        dataset = datasets.load_iris()
        rng = check_random_state(0)

        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.2,
                                                            random_state=rng)

        bounds = ([4.3, 2.0, 1.0, 0.1], [7.9, 4.4, 6.9, 2.5])

        clf_dp = GaussianNB(epsilon=1.0, bounds=bounds, random_state=rng)
        clf_non_private = sk_nb()

        for clf in [clf_dp, clf_non_private]:
            clf.fit(x_train, y_train)

        # Todo: remove try...except when sklearn v1.0 is required
        try:
            nonprivate_var = clf_non_private.var_
        except AttributeError:
            nonprivate_var = clf_non_private.sigma_

        theta_diff = (clf_dp.theta_ - clf_non_private.theta_) ** 2
        self.assertGreater(theta_diff.sum(), 0)

        var_diff = (clf_dp.var_ - nonprivate_var) ** 2
        self.assertGreater(var_diff.sum(), 0)

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_with_iris(self):
        from sklearn import datasets
        dataset = datasets.load_iris()
        rng = check_random_state(0)

        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.2,
                                                            random_state=rng)

        bounds = ([4.3, 2.0, 1.0, 0.1], [7.9, 4.4, 6.9, 2.5])

        clf = GaussianNB(epsilon=5.0, bounds=bounds, random_state=rng)
        clf.fit(x_train, y_train)

        accuracy = clf.score(x_test, y_test)
        counts = clf.class_count_.copy()
        self.assertGreater(accuracy, 0.45)

        clf.partial_fit(x_train, y_train)
        new_counts = clf.class_count_
        self.assertEqual(np.sum(new_counts), np.sum(counts) * 2)

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant()

        clf = GaussianNB(epsilon=1.0, bounds=(-2, 2), accountant=acc)
        clf.fit(X, y)
        self.assertEqual((1, 0), acc.total())

        with BudgetAccountant(1.5, 0) as acc2:
            clf = GaussianNB(epsilon=1.0, bounds=(-2, 2))
            clf.fit(X, y)
            self.assertEqual((1, 0), acc2.total())

            with self.assertRaises(BudgetError):
                clf.fit(X, y)

    def test_priors(self):
        clf = GaussianNB(epsilon=1, bounds=(-2, 2), priors=(0.75, 0.25))
        self.assertIsNotNone(clf.fit(X, y))

        clf = GaussianNB(epsilon=1, bounds=(-2, 2), priors=(1,))
        with self.assertRaises(ValueError):
            clf.fit(X, y)

        clf = GaussianNB(epsilon=1, bounds=(-2, 2), priors=(0.5, 0.7))
        with self.assertRaises(ValueError):
            clf.fit(X, y)

        clf = GaussianNB(epsilon=1, bounds=(-2, 2), priors=(-0.5, 1.5))
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_bad_refit_shape(self):
        clf = GaussianNB(epsilon=1, bounds=(-2, 2))
        clf.fit(X, y)

        X2 = np.random.random((6, 3))
        clf.bounds = ([0, 0, 0], [1, 1, 1])

        with self.assertRaises(ValueError):
            clf.partial_fit(X2, y)

    def test_bad_refit_classes(self):
        clf = GaussianNB(epsilon=1, bounds=(-2, 2))
        clf.fit(X, y)

        y2 = np.array([1, 1, 1, 2, 2, 3])

        with self.assertRaises(ValueError):
            clf.partial_fit(X, y2)

    def test_update_mean_variance(self):
        clf = GaussianNB(epsilon=1, bounds=([-2, -2], [2, 2]))
        self.assertIsNotNone(clf._update_mean_variance(0, 0, 0, X, None, n_noisy=5))
        self.assertIsNotNone(clf._update_mean_variance(0, 0, 0, X, None, n_noisy=0))
        self.assertWarns(PrivacyLeakWarning, clf._update_mean_variance, 0, 0, 0, X, None)
        self.assertWarns(DiffprivlibCompatibilityWarning, clf._update_mean_variance, 0, 0, 0, X, None, n_noisy=1,
                         sample_weight=1)

    def test_sigma(self):
        clf = GaussianNB(epsilon=1, bounds=(-2, 2))
        clf.fit(X, y)
        self.assertIsInstance(clf.sigma_, np.ndarray)

    def test_random_state(self):
        clf0 = GaussianNB(epsilon=1, bounds=(-2, 2), random_state=0)
        clf1 = GaussianNB(epsilon=1, bounds=(-2, 2), random_state=1)
        clf0.fit(X, y)
        clf1.fit(X, y)
        self.assertFalse(np.all(clf0.theta_ == clf1.theta_))
        self.assertFalse(np.all(clf0.var_ == clf1.var_))

        clf1 = GaussianNB(epsilon=1, bounds=(-2, 2), random_state=0)
        clf1.fit(X, y)
        self.assertTrue(np.all(clf0.theta_ == clf1.theta_))
        self.assertTrue(np.all(clf0.var_ == clf1.var_))
