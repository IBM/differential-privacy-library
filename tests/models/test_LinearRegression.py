import numpy as np
from unittest import TestCase

from diffprivlib.models.linear_regression import LinearRegression
from diffprivlib.utils import global_seed, PrivacyLeakWarning, DiffprivlibCompatibilityWarning


class TestLinearRegression(TestCase):
    def setup_method(self, method):
        global_seed(3141592653)

    def test_not_none(self):
        self.assertIsNotNone(LinearRegression)

    def test_no_params(self):
        clf = LinearRegression()

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

    def test_sample_weight_warning(self):
        clf = LinearRegression(data_norm=5.5, range_X=5, range_y=1)

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        with self.assertWarns(DiffprivlibCompatibilityWarning):
            clf.fit(X, y, sample_weight=np.ones_like(y))

    def test_no_range_y(self):
        clf = LinearRegression(data_norm=5.5, range_X=5)

        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

    def test_large_norm(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        clf = LinearRegression(data_norm=1.0)

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X, y)

    def test_different_results(self):
        from sklearn import datasets
        from sklearn import linear_model
        from sklearn.model_selection import train_test_split

        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

        clf = LinearRegression(data_norm=12, range_X=[3.6, 2.2, 5.9, 2.4], range_y=2)
        clf.fit(X_train, y_train)

        predict1 = clf.predict(X_test)

        clf = LinearRegression(data_norm=12, range_X=[3.6, 2.2, 5.9, 2.4], range_y=2)
        clf.fit(X_train, y_train)

        predict2 = clf.predict(X_test)

        clf = linear_model.LinearRegression()
        clf.fit(X_train, y_train)

        predict3 = clf.predict(X_test)

        self.assertFalse(np.all(predict1 == predict2))
        self.assertFalse(np.all(predict3 == predict1) and np.all(predict3 == predict2))

    def test_same_results(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn import linear_model

        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

        clf = LinearRegression(data_norm=12, epsilon=float("inf"), range_X=[3.6, 2.2, 5.9, 2.4], range_y=2)
        clf.fit(X_train, y_train)

        predict1 = clf.predict(X_test)

        clf = linear_model.LinearRegression(normalize=False)
        clf.fit(X_train, y_train)

        predict2 = clf.predict(X_test)

        self.assertTrue(np.allclose(predict1, predict2))

    def test_simple(self):
        X = np.linspace(-1, 1, 1000)
        y = X.copy()
        X = X[:, np.newaxis]

        clf = LinearRegression(epsilon=2, data_norm=1, range_X=[1], range_y=1, fit_intercept=False)
        clf.fit(X, y)

        self.assertIsNotNone(clf)
        self.assertAlmostEqual(clf.predict(np.array([0.5]).reshape(-1, 1))[0], 0.5, places=3)
