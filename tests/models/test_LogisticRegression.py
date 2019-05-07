import numpy as np
from unittest import TestCase

from diffprivlib.models.logistic_regression import LogisticRegression
from diffprivlib.utils import global_seed


class TestLogisticRegression(TestCase):
    def setup_method(self, method):
        global_seed(3141592653)

    def test_not_none(self):
        self.assertIsNotNone(LogisticRegression)

    def test_large_norm(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        clf = LogisticRegression(epsilon=10)

        with self.assertWarns(RuntimeWarning):
            clf.fit(X, y)

    def test_simple(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]
        X /= 6

        clf = LogisticRegression(epsilon=10)
        clf.fit(X, y)

        # print(clf.predict(np.array([0.5, 2, 5.5])))

        self.assertIsNotNone(clf)
        self.assertFalse(clf.predict(np.array([0.5 / 6])))
        self.assertTrue(clf.predict(np.array([5.5 / 6])))
