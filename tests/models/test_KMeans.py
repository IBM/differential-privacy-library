import numpy as np
from unittest import TestCase

from diffprivlib.models.k_means import KMeans


class TestKMeans(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(KMeans)

    def test_simple(self):
        clf = KMeans(1, [(0, 1)], 3)

        X = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
        centers = clf.fit(X)
        self.assertIn(0.1, centers)
        self.assertIn(0.5, centers)
        self.assertIn(0.9, centers)

    def test_predict(self):
        clf = KMeans(1, [(0, 1)], 3)

        X = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9])
        clf.fit(X)
        predicted = clf.predict(np.array([0.1, 0.5, 0.9]))
        self.assertNotEqual(predicted[0], predicted[1])
        self.assertNotEqual(predicted[0], predicted[2])
        self.assertNotEqual(predicted[2], predicted[1])
