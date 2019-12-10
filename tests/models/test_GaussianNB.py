from unittest import TestCase

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from diffprivlib.models.naive_bayes import GaussianNB
from diffprivlib.utils import global_seed, PrivacyLeakWarning, DiffprivlibCompatibilityWarning


class TestGaussianNB(TestCase):
    def test_not_none(self):
        clf = GaussianNB(epsilon=1, bounds=[(0, 1)])
        self.assertIsNotNone(clf)

    def test_zero_epsilon(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=0, bounds=[(0, 1)])

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_neg_epsilon(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=-1, bounds=[(0, 1)])

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_sample_weight_warning(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=1, bounds=[(0, 1), (0, 1)])
        w = abs(np.random.randn(10))

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            clf.fit(X, y, sample_weight=w)

    def test_mis_ordered_bounds(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=[(0, 1), (1, 0)])

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_no_bounds(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB()

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

        self.assertIsNotNone(clf)

    def test_missing_bounds(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=[(0, 1)])

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_different_results(self):
        from sklearn.naive_bayes import GaussianNB as sk_nb
        from sklearn import datasets

        global_seed(12345)
        dataset = datasets.load_iris()

        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.2)

        bounds = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]

        clf_dp = GaussianNB(epsilon=1.0, bounds=bounds)
        clf_non_private = sk_nb()

        for clf in [clf_dp, clf_non_private]:
            clf.fit(x_train, y_train)

        same_prediction = clf_dp.predict(x_test) == clf_non_private.predict(x_test)

        self.assertFalse(np.all(same_prediction))

    def test_with_iris(self):
        global_seed(12345)
        from sklearn import datasets
        dataset = datasets.load_iris()

        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.2)

        bounds = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]

        clf = GaussianNB(epsilon=1.0, bounds=bounds)
        clf.fit(x_train, y_train)

        accuracy = sum(clf.predict(x_test) == y_test) / y_test.shape[0]
        # print(accuracy)
        self.assertGreater(accuracy, 0.5)
