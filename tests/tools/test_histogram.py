import numpy as np
from unittest import TestCase

from diffprivlib.tools.histograms import histogram
from diffprivlib.utils import global_seed


class TestHistogram(TestCase):
    def test_no_params(self):
        a = np.array([1, 2, 3, 4, 5])
        with self.assertWarns(RuntimeWarning):
            res = histogram(a)
        self.assertIsNotNone(res)

    def test_no_range(self):
        a = np.array([1, 2, 3, 4, 5])
        with self.assertWarns(RuntimeWarning):
            res = histogram(a, epsilon=1)
        self.assertIsNotNone(res)

    def test_same_edges(self):
        a = np.array([1, 2, 3, 4, 5])
        _, edges = np.histogram(a, bins=3, range=(0, 10))
        _, dp_edges = histogram(a, epsilon=1, bins=3, range=(0, 10))
        self.assertTrue((edges == dp_edges).all())

    def test_different_result(self):
        global_seed(3141592653)
        a = np.array([1, 2, 3, 4, 5])
        hist, _ = np.histogram(a, bins=3, range=(0, 10))
        dp_hist, _ = histogram(a, epsilon=0.1, bins=3, range=(0, 10))

        # print("Non-private histogram: %s" % hist)
        # print("Private histogram: %s" % dp_hist)
        self.assertTrue((hist != dp_hist).any())

    def test_density(self):
        global_seed(3141592653)
        a = np.array([1, 2, 3, 4, 5])
        dp_hist, _ = histogram(a, epsilon=0.1, bins=3, range=(0, 10), density=True)

        # print(dp_hist.sum())

        self.assertAlmostEqual(dp_hist.sum(), 1.0 * 3 / 10)
