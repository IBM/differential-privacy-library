import numpy as np
from unittest import TestCase

from diffprivlib.tools.histograms import histogramdd
from diffprivlib.utils import global_seed, PrivacyLeakWarning


class TestHistogramdd(TestCase):
    def test_no_params(self):
        a = np.array([1, 2, 3, 4, 5])
        with self.assertWarns(PrivacyLeakWarning):
            res = histogramdd(a)
        self.assertIsNotNone(res)

    def test_no_range(self):
        a = np.array([1, 2, 3, 4, 5])
        with self.assertWarns(PrivacyLeakWarning):
            res = histogramdd(a, epsilon=2)
        self.assertIsNotNone(res)

    def test_same_edges(self):
        a = np.array([1, 2, 3, 4, 5])
        _, edges = np.histogramdd(a, bins=3, range=[(0, 10)])
        _, dp_edges = histogramdd(a, epsilon=1, bins=3, range=[(0, 10)])

        for i in range(len(edges)):
            self.assertTrue((edges[i] == dp_edges[i]).all())

    def test_different_result(self):
        global_seed(3141592653)
        a = np.array([1, 2, 3, 4, 5])
        hist, _ = np.histogramdd(a, bins=3, range=[(0, 10)])
        dp_hist, _ = histogramdd(a, epsilon=0.1, bins=3, range=[(0, 10)])

        # print("Non-private histogram: %s" % hist)
        # print("Private histogram: %s" % dp_hist)
        self.assertTrue((hist != dp_hist).any())

    def test_density_1d(self):
        global_seed(3141592653)
        a = np.array([1, 2, 3, 4, 5])
        dp_hist, _ = histogramdd(a, epsilon=1, bins=3, range=[(0, 10)], density=True)

        # print(dp_hist.sum())

        self.assertAlmostEqual(dp_hist.sum(), 1.0 * 3 / 10)

    def test_density_2d(self):
        global_seed(3141592653)
        a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T
        dp_hist, _ = histogramdd(a, epsilon=1, bins=3, range=[(0, 10), (0, 10)], density=True)

        # print(dp_hist.sum())

        self.assertAlmostEqual(dp_hist.sum(), 1.0 * (3 / 10) ** 2)
