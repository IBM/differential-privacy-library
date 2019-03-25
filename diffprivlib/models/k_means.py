import numpy as np

from sklearn.base import BaseEstimator


# noinspection PyPep8Naming
class KMeans(BaseEstimator):
    def __init__(self, epsilon, bounds, n_clusters=8, verbose=0):
        self.epsilon = epsilon
        self.bounds = bounds
        self.bounds_processed = None
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.fitted_centers = None

    def fit(self, X, y=None, sample_weight=None):
        del y, sample_weight
        iters = 7

        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)

        dims = X.shape[1]

        if len(self.bounds) != dims:
            raise ValueError("Number of dimensions of X must match number of bounds")

        centers = self._init_centers(dims)

        for i in range(iters):
            distances, labels = self._distances_labels(X, centers)

            centers = self._update_centers(X, centers, labels)

        self.fitted_centers = centers

        return centers

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X)
        return self.predict(X)

    def predict(self, X, sample_weight=None):
        if self.fitted_centers is None:
            raise ValueError("Classifier not fitted yet. Run `.fit()` with training data first.")

        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)

        return self._distances_labels(X, self.fitted_centers)[1]

    def _init_centers(self, dims):

        if self.bounds_processed is None:
            bounds_processed = np.zeros(shape=(dims, 2))

            for dim in range(dims):
                lower = self.bounds[dim][0]
                upper = self.bounds[dim][1]

                bounds_processed[dim, :] = [upper - lower, lower]

            self.bounds_processed = bounds_processed

        # Find clusters at least cluster_proximity apart
        cluster_proximity = np.min(self.bounds_processed[:, 0]) / 2.0

        while cluster_proximity > 0:
            centers = np.zeros(shape=(self.n_clusters, dims))
            cluster, retry = 0, 0

            while retry < 100:
                if cluster >= self.n_clusters:
                    break

                temp_center = np.random.random(dims) * self.bounds_processed[:, 0] + self.bounds_processed[:, 1]

                if cluster == 0:
                    centers[0, :] = temp_center
                    cluster += 1
                    continue

                min_distance = ((centers[:cluster, :] - temp_center) ** 2).sum(axis=1).min()

                if np.sqrt(min_distance) >= cluster_proximity:
                    centers[cluster, :] = temp_center
                    cluster += 1
                else:
                    retry += 1

            if cluster >= self.n_clusters:
                return centers
            else:
                cluster_proximity /= 2.0

        return None

    def _distances_labels(self, X, centers):
        distances = np.zeros((X.shape[0], self.n_clusters))

        for cluster in range(self.n_clusters):
            distances[:, cluster] = ((X - centers[cluster, :])**2).sum(axis=1)

        labels = np.argmin(distances, axis=1)
        return distances, labels

    def _update_centers(self, X, centers, labels):
        for cluster in range(self.n_clusters):
            if cluster not in labels:
                continue

            temp = np.mean(X[labels == cluster], axis=0)
            centers[cluster, :] = temp

        return centers
