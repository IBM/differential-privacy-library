from unittest import TestCase

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from diffprivlib.models.naive_bayes import GaussianNB
from diffprivlib.utils import global_seed, PrivacyLeakWarning, DiffprivlibCompatibilityWarning, BudgetError


class TestGaussianNB(TestCase):
    def test_not_none(self):
        clf = GaussianNB(epsilon=1, bounds=(0, 1))
        self.assertIsNotNone(clf)

    def test_zero_epsilon(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=0, bounds=(0, 1))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_neg_epsilon(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=-1, bounds=(0, 1))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_sample_weight_warning(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)
        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))
        w = abs(np.random.randn(10))

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            clf.fit(X, y, sample_weight=w)

    def test_mis_ordered_bounds(self):
        X = np.random.random((10, 2))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 1], [1, 0]))

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
        X = np.random.random((10, 3))
        y = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1, bounds=([0, 0], [1, 1]))

        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_noisy_count(self):
        y = np.random.randint(20, size=10000)
        actual_counts = np.array([(y == y_i).sum() for y_i in np.unique(y)])

        clf = GaussianNB(epsilon=3)
        noisy_counts = clf._noisy_class_counts(y)
        self.assertEqual(y.shape[0], noisy_counts.sum())
        self.assertFalse(np.all(noisy_counts == actual_counts))

        clf = GaussianNB(epsilon=float("inf"))
        noisy_counts = clf._noisy_class_counts(y)
        self.assertEqual(y.shape[0], noisy_counts.sum())
        self.assertTrue(np.all(noisy_counts == actual_counts))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_different_results(self):
        from sklearn.naive_bayes import GaussianNB as sk_nb
        from sklearn import datasets

        global_seed(12345)
        dataset = datasets.load_iris()

        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.2)

        bounds = ([4.3, 2.0, 1.0, 0.1], [7.9, 4.4, 6.9, 2.5])

        clf_dp = GaussianNB(epsilon=1.0, bounds=bounds)
        clf_non_private = sk_nb()

        for clf in [clf_dp, clf_non_private]:
            clf.fit(x_train, y_train)

        same_prediction = clf_dp.predict(x_test) == clf_non_private.predict(x_test)

        self.assertFalse(np.all(same_prediction))

    @pytest.mark.filterwarnings('ignore: numpy.ufunc size changed')
    def test_with_iris(self):
        global_seed(12345)
        from sklearn import datasets
        dataset = datasets.load_iris()

        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=.2)

        bounds = ([4.3, 2.0, 1.0, 0.1], [7.9, 4.4, 6.9, 2.5])

        clf = GaussianNB(epsilon=1.0, bounds=bounds)
        clf.fit(x_train, y_train)

        accuracy = clf.score(x_test, y_test)
        counts = clf.class_count_.copy()
        # print(accuracy)
        self.assertGreater(accuracy, 0.5)

        clf.partial_fit(x_train, y_train)
        new_counts = clf.class_count_
        self.assertEqual(np.sum(new_counts), np.sum(counts) * 2)

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant
        acc = BudgetAccountant()

        x_train = np.random.random((10, 2))
        y_train = np.random.randint(2, size=10)

        clf = GaussianNB(epsilon=1.0, bounds=(0, 1), accountant=acc)
        clf.fit(x_train, y_train)
        self.assertEqual((1, 0), acc.total())

        with BudgetAccountant(1.5, 0) as acc2:
            clf = GaussianNB(epsilon=1.0, bounds=(0, 1))
            clf.fit(x_train, y_train)
            self.assertEqual((1, 0), acc2.total())

            with self.assertRaises(BudgetError):
                clf.fit(x_train, y_train)
