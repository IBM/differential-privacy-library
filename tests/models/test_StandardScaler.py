from unittest import TestCase

import numpy as np
from sklearn.utils.extmath import _incremental_mean_and_var as sk_incremental_mean_and_var
import sklearn.preprocessing as sk_pp

from diffprivlib.models.standard_scaler import _incremental_mean_and_var, StandardScaler
from diffprivlib.utils import PrivacyLeakWarning, global_seed


class TestIncrementalMeanAndVar(TestCase):
    def test_no_range(self):
        X = np.random.rand(5, 10)
        with self.assertWarns(PrivacyLeakWarning):
            _incremental_mean_and_var(X, epsilon=float("inf"), range=None, last_mean=0., last_variance=None,
                                      last_sample_count=0)

    def test_inf_epsilon(self):
        X = np.random.rand(5, 10)
        dp_mean, dp_var, dp_count = _incremental_mean_and_var(X, epsilon=float("inf"), range=1, last_mean=0.,
                                                              last_variance=None,
                                                              last_sample_count=np.zeros(X.shape[1], dtype=np.int64))
        sk_mean, sk_var, sk_count = sk_incremental_mean_and_var(X, last_mean=0., last_variance=None,
                                                                last_sample_count=np.zeros(X.shape[1], dtype=np.int64))

        self.assertTrue(np.allclose(dp_mean, sk_mean))
        self.assertIsNone(dp_var)
        self.assertIsNone(sk_var)
        self.assertTrue((dp_count == sk_count).all())

        dp_mean, dp_var, dp_count = _incremental_mean_and_var(X, epsilon=float("inf"), range=1, last_mean=0.,
                                                              last_variance=0.,
                                                              last_sample_count=np.zeros(X.shape[1], dtype=np.int64))
        sk_mean, sk_var, sk_count = sk_incremental_mean_and_var(X, last_mean=0., last_variance=0.,
                                                                last_sample_count=np.zeros(X.shape[1], dtype=np.int64))

        self.assertTrue(np.allclose(dp_mean, sk_mean))
        self.assertTrue(np.allclose(dp_var, sk_var))
        self.assertTrue((dp_count == sk_count).all())

    def test_increment_inf_epsilon(self):
        X = np.ones((5, 1))
        dp_mean, dp_var, dp_count = _incremental_mean_and_var(X, epsilon=float("inf"), range=1, last_mean=0.,
                                                              last_variance=None, last_sample_count=5)
        self.assertAlmostEqual(dp_mean, 0.5, places=5)
        self.assertEqual(dp_count, 10)

    def test_duplicate_dataset(self):
        X = np.random.rand(10, 5)
        mean1, var1, count1 = _incremental_mean_and_var(X, epsilon=float("inf"), range=1, last_mean=0.,
                                                        last_variance=0., last_sample_count=0)

        mean2, var2, count2 = _incremental_mean_and_var(X, epsilon=float("inf"), range=1, last_mean=mean1,
                                                        last_variance=var1, last_sample_count=count1)

        self.assertTrue(np.allclose(mean1, mean2))
        self.assertTrue(np.allclose(var1, var2))
        self.assertTrue(np.all(count1 == 10), "Counts should be 10, got %s" % count1)
        self.assertTrue(np.all(count2 == 20), "Counts should be 20, got %s" % count2)

    def test_different_results(self):
        X = np.random.rand(10, 5)
        mean1, var1, count1 = _incremental_mean_and_var(X, epsilon=1, range=1, last_mean=0., last_variance=0.,
                                                        last_sample_count=0)

        mean2, var2, count2 = _incremental_mean_and_var(X, epsilon=1, range=1, last_mean=0, last_variance=0,
                                                        last_sample_count=0)

        self.assertFalse(np.allclose(mean1, mean2, atol=1e-2))
        self.assertFalse(np.allclose(var1, var2, atol=1e-2))
        self.assertTrue(np.all(count1 == 10), "Counts should be 10, got %s" % count1)
        self.assertTrue(np.all(count2 == 10), "Counts should be 10, got %s" % count2)


class TestStandardScaler(TestCase):
    def test_class(self):
        from sklearn.base import BaseEstimator
        self.assertTrue(issubclass(StandardScaler, BaseEstimator))

    def test_not_none(self):
        ss = StandardScaler()
        self.assertIsNotNone(ss)

    def test_no_range(self):
        X = np.random.rand(10, 5)
        ss = StandardScaler()

        with self.assertWarns(PrivacyLeakWarning):
            ss.fit(X)

    def test_inf_epsilon(self):
        X = np.random.rand(10, 5)

        dp_ss = StandardScaler(range=1, epsilon=float("inf"))
        dp_ss.fit(X)

        sk_ss = sk_pp.StandardScaler()
        sk_ss.fit(X)

        self.assertTrue(np.allclose(dp_ss.mean_, sk_ss.mean_), "Arrays %s and %s should be the same" %
                        (dp_ss.mean_, sk_ss.mean_))
        self.assertTrue(np.allclose(dp_ss.var_, sk_ss.var_), "Arrays %s and %s should be the same" %
                        (dp_ss.var_, sk_ss.var_))
        self.assertTrue(np.all(dp_ss.n_samples_seen_ == sk_ss.n_samples_seen_))

    def test_different_results(self):
        X = np.random.rand(10, 5)

        ss1 = StandardScaler(range=1)
        ss1.fit(X)

        ss2 = StandardScaler(range=1)
        ss2.fit(X)

        self.assertFalse(np.allclose(ss1.mean_, ss2.mean_), "Arrays %s and %s should be the same" %
                        (ss1.mean_, ss2.mean_))
        self.assertFalse(np.allclose(ss1.var_, ss2.var_), "Arrays %s and %s should be the same" %
                        (ss1.var_, ss2.var_))

    def test_functionality(self):
        X = np.random.rand(10, 5)
        ss = StandardScaler(range=1)
        ss.fit(X)

        self.assertIsNotNone(ss.transform(X))
        self.assertIsNotNone(ss.inverse_transform(X))
        self.assertIsNotNone(ss.fit_transform(X))

    def test_similar_results(self):
        global_seed(314159)

        X = np.random.rand(100000, 5)

        dp_ss = StandardScaler(range=1, epsilon=float("inf"))
        dp_ss.fit(X)

        sk_ss = sk_pp.StandardScaler()
        sk_ss.fit(X)

        self.assertTrue(np.allclose(dp_ss.mean_, sk_ss.mean_, rtol=1, atol=1e-4), "Arrays %s and %s should be close" %
                        (dp_ss.mean_, sk_ss.mean_))
        self.assertTrue(np.allclose(dp_ss.var_, sk_ss.var_, rtol=1, atol=1e-4), "Arrays %s and %s should be close" %
                        (dp_ss.var_, sk_ss.var_))
        self.assertTrue(np.all(dp_ss.n_samples_seen_ == sk_ss.n_samples_seen_))
