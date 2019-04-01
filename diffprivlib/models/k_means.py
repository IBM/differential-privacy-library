import numpy as np

from sklearn.base import BaseEstimator


# noinspection PyPep8Naming
from diffprivlib.mechanisms import LaplaceBoundedDomain, GeometricFolded


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

            centers = self._update_centers(X, centers, labels, dims)

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
                    retry = 0
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

    def _update_centers(self, X, centers, labels, dims):
        epsilon_0, epsilon_i = self._split_epsilon(dims)
        # laplace_mechs = []
        geometric_mech = GeometricFolded().set_sensitivity(1).set_bounds(0.5, float("inf")).set_epsilon(epsilon_0)
        laplace_mech = LaplaceBoundedDomain().set_epsilon(epsilon_i)

        # for i in range(dims):
        #     laplace_mechs.append(LaplaceBoundedDomain()
        #                          .set_sensitivity(self.bounds[i][1] - self.bounds[i][0])
        #                          .set_bounds(self.bounds[i][0], self.bounds[i][1])
        #                          .set_epsilon(epsilon_i))

        for cluster in range(self.n_clusters):
            if cluster not in labels:
                continue

            cluster_count = sum(labels == cluster)
            noisy_count = geometric_mech.randomise(cluster_count)

            cluster_sum = np.sum(X[labels == cluster], axis=0)
            noisy_sum = np.zeros_like(cluster_sum)

            for i in range(dims):
                _mech = laplace_mech.copy().set_sensitivity(self.bounds[i][1] - self.bounds[i][0])\
                    .set_bounds(noisy_count * self.bounds[i][0], noisy_count * self.bounds[i][1])
                noisy_sum[i] = _mech.randomise(cluster_sum[i])

            centers[cluster, :] = noisy_sum / noisy_count

        return centers

    def _split_epsilon(self, dims, rho=0.225):
        """
        Split epsilon between sum perturbation and count perturbation, as proposed by Su et al.

        Paper link: http://delivery.acm.org/10.1145/2860000/2857708/p26-su.pdf
        :param dims: Number of dimensions to split epsilon between
        :type dims: int
        :param rho: Coordinate normalisation factor, default 0.225
        :type rho: float
        :return: (epsilon_0, epsilon_i)
        """

        epsilon_i = 1
        epsilon_0 = np.cbrt(4 * dims * rho ** 2)

        normaliser = self.epsilon / (epsilon_i * dims + epsilon_0)

        return epsilon_i * normaliser, epsilon_0 * normaliser
