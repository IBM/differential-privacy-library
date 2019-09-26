from unittest import TestCase

import numpy as np

from diffprivlib.models.logistic_regression import _logistic_regression_path


class TestLogisticRegression(TestCase):
    def setup_class(self):
        X = np.array(
            [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
             5.00, 5.50])
        y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
        X = X[:, np.newaxis]

        self.X = X
        self.y = y

    def test_not_none(self):
        self.assertIsNotNone(_logistic_regression_path)

    def test_with_dataset(self):
        output = _logistic_regression_path(self.X, self.y, Cs=[1e5])

        self.assertIsInstance(output, tuple)
