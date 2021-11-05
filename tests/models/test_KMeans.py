import numpy as np
from unittest import TestCase

from diffprivlib.models.k_means import KMeans
from diffprivlib.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning, BudgetError


class TestKMeans(TestCase):
    def test_not_none(self):
        self.assertIsNotNone(KMeans)

    def test_simple(self):
        clf = KMeans(3, epsilon=5, bounds=(0, 1), random_state=0)

        X = np.zeros(600) + 0.1
        X[:400] = 0.5
        X[:200] = 0.9
        X = X.reshape(-1, 1)

        clf.fit(X)
        centers = clf.cluster_centers_

        self.assertAlmostEqual(np.min(centers), 0.1, delta=0.1)
        self.assertAlmostEqual(np.median(centers), 0.5, delta=0.1)
        self.assertAlmostEqual(np.max(centers), 0.9, delta=0.1)

    def test_unused_args(self):
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            KMeans(verbose=1)

    def test_1d_array(self):
        clf = KMeans()

        X = np.zeros(10)

        with self.assertRaises(ValueError):
            clf.fit(X)

    def test_no_bounds(self):
        clf = KMeans()

        X = np.zeros(10).reshape(-1, 1)

        with self.assertWarns(PrivacyLeakWarning):
            clf.fit(X)

    def test_predict(self):
        clf = KMeans(3, epsilon=1, bounds=(0, 1), random_state=0)

        X = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]).reshape(-1, 1)
        clf.fit(X)
        predicted = clf.predict(np.array([0.1, 0.5, 0.9]).reshape(-1, 1))

        # print("Predictions: %s" % str(predicted))
        # print("Centers: %s" % str(clf.cluster_centers_))

        self.assertTrue(0 <= predicted[0] <= 2)
        self.assertTrue(0 <= predicted[1] <= 2)
        self.assertTrue(0 <= predicted[2] <= 2)

    def test_sample_weights(self):
        clf = KMeans(3, epsilon=30, bounds=(0, 1))

        X = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]).reshape(-1, 1)
        with self.assertWarns(DiffprivlibCompatibilityWarning):
            clf.fit(X, None, 1)

    def test_too_many_clusters(self):
        clf = KMeans(100, epsilon=30, bounds=(0, 1))

        X = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]).reshape(-1, 1)
        self.assertRaises(ValueError, clf.fit, X)

    def test_inf_epsilon(self):
        clf = KMeans(3, epsilon=float("inf"), bounds=(0, 1), random_state=0)

        X = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9] * 3).reshape(-1, 1)
        clf.fit(X)
        centers = clf.cluster_centers_.flatten()

        self.assertTrue(np.isclose(0.1, centers).any())
        self.assertTrue(np.isclose(0.5, centers).any())
        self.assertTrue(np.isclose(0.9, centers).any())

    def test_many_features(self):
        rng = np.random.RandomState(0)
        X = rng.random(size=(500, 3))
        bounds = (0, 1)

        clf = KMeans(4, bounds=bounds, random_state=rng)

        clf.fit(X)
        centers = clf.cluster_centers_

        self.assertEqual(centers.shape[0], 4)
        self.assertEqual(centers.shape[1], 3)

        self.assertTrue(np.all(clf.transform(X).argmin(axis=1) == clf.predict(X)))

    def test_random_state(self):
        rng = np.random.RandomState(0)
        X = rng.random(size=(500, 3))
        bounds = (0, 1)

        clf0 = KMeans(4, bounds=bounds, random_state=0)
        clf1 = KMeans(4, bounds=bounds, random_state=1)
        clf0.fit(X)
        clf1.fit(X)
        self.assertFalse(np.any(clf0.cluster_centers_ == clf1.cluster_centers_))

        clf1 = KMeans(4, bounds=bounds, random_state=0)
        clf1.fit(X)
        self.assertTrue(np.all(clf0.cluster_centers_ == clf1.cluster_centers_))

    def test_accountant(self):
        from diffprivlib.accountant import BudgetAccountant

        acc = BudgetAccountant()
        clf = KMeans(3, epsilon=30, bounds=(0, 1), accountant=acc)

        X = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]).reshape(-1, 1)
        clf.fit(X)
        self.assertEqual((30, 0), acc.total())

        clf.fit(X)
        self.assertEqual((60, 0), acc.total())

        with BudgetAccountant(15, 0) as acc2:
            clf2 = KMeans(3, epsilon=10, bounds=(0, 1))
            clf2.fit(X)
            self.assertEqual((10, 0), acc2.total())

            with self.assertRaises(BudgetError):
                clf2.fit(X)
