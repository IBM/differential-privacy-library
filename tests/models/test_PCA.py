from unittest import TestCase

import numpy as np
import pytest
from sklearn.model_selection import train_test_split
import sklearn.decomposition.pca as sk_pca

from diffprivlib.models.pca import PCA
from diffprivlib.utils import global_seed, PrivacyLeakWarning


class TestGaussianNB(TestCase):
    def test_not_none(self):
        clf = PCA(epsilon=1, centered=True, data_norm=1)
        self.assertIsNotNone(clf)

    def test_inf_epsilon(self):
        X = np.random.randn(250, 21)
        X -= np.mean(X, axis=0)

        for components in range(1, 10):
            clf = PCA(components, epsilon=float("inf"), centered=True, data_norm=1)
            clf.fit(X)

            sk_clf = sk_pca.PCA(components, svd_solver='full')
            sk_clf.fit(X)

            self.assertAlmostEqual(clf.score(X), sk_clf.score(X))
            self.assertTrue(np.allclose(clf.get_precision(), sk_clf.get_precision()))
            self.assertTrue(np.allclose(clf.get_covariance(), sk_clf.get_covariance()))
            self.assertTrue(np.allclose(np.abs((clf.components_ / sk_clf.components_).sum(axis=1)), clf.n_features_))
